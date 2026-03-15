import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import random

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType

# 复用train.py中的类定义
class TextClassificationDataset(Dataset):
    """用于文本分类的数据集类"""
    def __init__(self, data_list, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data_list
    
    def _encode_item(self, item):
        """将单个样本编码为模型输入"""
        if "conversations" in item:
            parts = []
            for turn in item["conversations"]:
                role = "User" if turn["role"] == "user" else "Assistant"
                parts.append(f"{role}: {turn['content']}")
            text = "\n\n".join(parts)
        elif "text" in item:
            text = item["text"]
        else:
            text = f"Question: {item['instruction']}\nAnswer: {item['response']}"

        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        seq_length = attention_mask.sum().item()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "seq_length": torch.tensor(seq_length, dtype=torch.float32),
            "original_data": item
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return self._encode_item(item)


class LengthClassifier(nn.Module):
    """长度分类器"""
    def __init__(self, num_labels=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels)
        )
    
    def forward(self, seq_lengths):
        x = seq_lengths.unsqueeze(-1)
        logits = self.net(x)
        return logits


class ClassifierWithLoRAAndPoE(nn.Module):
    """使用LoRA的分类器模型，带有PoE机制"""
    def __init__(self, model_id, num_labels=2, use_lora=True, tokenizer=None, use_poe=True):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_id)
        self.num_labels = num_labels
        self.use_poe = use_poe
        
        if tokenizer is not None:
            self.config.pad_token_id = tokenizer.pad_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=self.config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        if tokenizer is not None:
            self.model.config.pad_token_id = tokenizer.pad_token_id
        
        if use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, lora_config)
        
        device = next(self.model.parameters()).device
        hidden_size = self.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels, dtype=torch.bfloat16).to(device)
        
        if self.use_poe:
            self.length_classifier = LengthClassifier(num_labels=num_labels).to(device)
        else:
            self.length_classifier = None
        
        self.loss_fn = nn.CrossEntropyLoss().to(device)
    
    def forward(self, input_ids, attention_mask, seq_lengths=None, labels=None):
        with torch.cuda.amp.autocast(enabled=True):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            hidden_states = outputs.hidden_states[-1]
            del outputs.hidden_states
        
        seq_lengths_from_mask = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        seq_lengths_from_mask = torch.clamp(seq_lengths_from_mask, max=hidden_states.shape[1]-1)
        
        pooled_output = torch.stack([hidden_states[i, seq_lengths_from_mask[i], :] for i in range(batch_size)])
        
        device = next(self.classifier.parameters()).device
        dtype = next(self.classifier.parameters()).dtype
        pooled_output = pooled_output.to(device=device, dtype=dtype)
        
        logits_main = self.classifier(pooled_output)
        
        if self.use_poe and seq_lengths is not None:
            seq_lengths = seq_lengths.to(device)
            logits_length = self.length_classifier(seq_lengths)
            logits = logits_main + logits_length
        else:
            logits = logits_main
        
        loss = None
        if labels is not None:
            labels = labels.to(device)
            loss = self.loss_fn(logits, labels)
        
        from types import SimpleNamespace
        return SimpleNamespace(
            loss=loss, 
            logits=logits,
            logits_main=logits_main,
            logits_length=logits_length if self.use_poe and seq_lengths is not None else None
        )


def collate_fn(batch):
    """将批次数据整合为模型输入格式"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    seq_lengths = torch.stack([item["seq_length"] for item in batch])
    original_data = [item["original_data"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "seq_length": seq_lengths,
        "original_data": original_data
    }


def load_jsonl(path):
    """加载JSONL文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def predict_probabilities(model, dataloader, device):
    """使用模型预测所有数据的概率"""
    model.eval()
    all_probs = []
    all_data = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="预测中"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            seq_lengths = batch["seq_length"].to(device)
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                seq_lengths=seq_lengths
            )
            logits = outputs.logits
            
            # 计算概率（softmax）
            probs = torch.softmax(logits, dim=1)
            # 获取预测为合成数据（label=1）的概率
            synthetic_probs = probs[:, 1].cpu().numpy()
            
            all_probs.extend(synthetic_probs)
            all_data.extend(batch["original_data"])
    
    return all_probs, all_data


def compute_distribution(probs, num_bins=10):
    """计算概率分布"""
    bins = np.linspace(0, 1, num_bins + 1)
    hist, _ = np.histogram(probs, bins=bins)
    distribution = hist / len(probs)
    return distribution, bins


def stratified_sample_by_distribution(data_with_probs, target_distribution, bins, total_samples):
    """根据目标分布进行分层采样"""
    # 将数据按照概率区间分组
    binned_data = defaultdict(list)
    for item, prob in data_with_probs:
        bin_idx = np.digitize(prob, bins) - 1
        bin_idx = min(bin_idx, len(bins) - 2)  # 确保不越界
        binned_data[bin_idx].append((item, prob))
    
    # 按目标分布采样
    sampled_data = []
    for bin_idx, target_ratio in enumerate(target_distribution):
        target_count = int(total_samples * target_ratio)
        available = binned_data[bin_idx]
        
        if len(available) < target_count:
            logging.warning(f"区间 {bin_idx} 数据不足: 需要 {target_count}, 实际 {len(available)}")
            sampled_data.extend(available)
        else:
            sampled = random.sample(available, target_count)
            sampled_data.extend(sampled)
    
    # 如果采样数量不足，从剩余数据中随机补充
    if len(sampled_data) < total_samples:
        remaining = total_samples - len(sampled_data)
        all_available = [item for bin_data in binned_data.values() for item in bin_data]
        already_sampled = set(id(item) for item, _ in sampled_data)
        not_sampled = [(item, prob) for item, prob in all_available if id(item) not in already_sampled]
        
        if not_sampled:
            additional = random.sample(not_sampled, min(remaining, len(not_sampled)))
            sampled_data.extend(additional)
    
    # 如果采样数量过多，随机删除多余的
    if len(sampled_data) > total_samples:
        sampled_data = random.sample(sampled_data, total_samples)
    
    return [item for item, _ in sampled_data]


def visualize_distributions(oasst2_dist, magpie_before_dist, magpie_after_dist, bins, save_path):
    """可视化分布对比"""
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.25
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：筛选前对比
    x = np.arange(len(bin_centers))
    ax1.bar(x - width, oasst2_dist, width, label='oasst2 (真实数据)', alpha=0.8)
    ax1.bar(x, magpie_before_dist, width, label='magpie 筛选前', alpha=0.8)
    ax1.set_xlabel('分类器预测概率区间（预测为合成数据的概率）')
    ax1.set_ylabel('样本比例')
    ax1.set_title('筛选前：oasst2 vs magpie 预测概率分布')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)], rotation=45)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 右图：筛选后对比
    ax2.bar(x - width/2, oasst2_dist, width, label='oasst2 (真实数据)', alpha=0.8)
    ax2.bar(x + width/2, magpie_after_dist, width, label='magpie 筛选后', alpha=0.8)
    ax2.set_xlabel('分类器预测概率区间（预测为合成数据的概率）')
    ax2.set_ylabel('样本比例')
    ax2.set_title('筛选后：oasst2 vs magpie 预测概率分布（已对齐）')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)], rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"分布对比图已保存至: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="使用训练好的分类器预测并筛选数据")
    parser.add_argument("--model_id", type=str, default="/home/share/models/Llama-3.2-1B", help="基础模型ID")
    parser.add_argument("--checkpoint", type=str, default="./data/experiments/exp001_multiturn_classifier/best_model.pt", help="模型检查点路径")
    parser.add_argument("--input_file", type=str, default="./data/train_round2.jsonl", help="输入数据文件")
    parser.add_argument("--output_file", type=str, default="./data/train_round2_filtered.jsonl", help="输出文件路径")
    parser.add_argument("--output_dir", type=str, default="./data/round2_filtering", help="输出目录（保存报告和图表）")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--max_length", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--num_bins", type=int, default=10, help="概率分布的区间数量")
    parser.add_argument("--target_samples", type=int, default=5000, help="每个来源的目标采样数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--use_poe", action="store_true", default=True, help="使用PoE机制")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.info("=" * 60)
    logging.info("第二轮数据筛选 - 基于分类器预测分布匹配")
    logging.info("=" * 60)
    
    # 1. 加载tokenizer
    logging.info(f"正在加载 Tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
    # 2. 加载模型
    logging.info(f"正在加载模型: {args.model_id}")
    model = ClassifierWithLoRAAndPoE(
        args.model_id,
        num_labels=2,
        use_lora=True,
        tokenizer=tokenizer,
        use_poe=args.use_poe
    )
    
    # 加载检查点
    logging.info(f"正在加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    device = next(model.parameters()).device
    logging.info(f"使用设备: {device}")
    
    # 3. 加载数据
    logging.info(f"正在加载数据: {args.input_file}")
    all_data = load_jsonl(args.input_file)
    logging.info(f"总数据量: {len(all_data)}")
    
    # 分离oasst2和magpie数据
    oasst2_data = [item for item in all_data if item.get("source") == "oasst2" or item.get("label") == 0]
    magpie_data = [item for item in all_data if item.get("source") == "magpie" or item.get("label") == 1]
    
    logging.info(f"oasst2 数据: {len(oasst2_data)} 条")
    logging.info(f"magpie 数据: {len(magpie_data)} 条")
    
    # 4. 创建数据集和数据加载器
    dataset = TextClassificationDataset(all_data, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    # 5. 预测所有数据的概率
    logging.info("开始预测...")
    all_probs, all_data_ordered = predict_probabilities(model, dataloader, device)
    
    # 6. 将概率与数据关联
    data_with_probs = list(zip(all_data_ordered, all_probs))
    
    # 按来源分组
    oasst2_with_probs = [(item, prob) for item, prob in data_with_probs if item.get("source") == "oasst2" or item.get("label") == 0]
    magpie_with_probs = [(item, prob) for item, prob in data_with_probs if item.get("source") == "magpie" or item.get("label") == 1]
    
    oasst2_probs = [prob for _, prob in oasst2_with_probs]
    magpie_probs = [prob for _, prob in magpie_with_probs]
    
    logging.info(f"\n预测概率统计:")
    logging.info(f"oasst2 平均预测概率: {np.mean(oasst2_probs):.4f} ± {np.std(oasst2_probs):.4f}")
    logging.info(f"magpie 平均预测概率: {np.mean(magpie_probs):.4f} ± {np.std(magpie_probs):.4f}")
    
    # 7. 计算oasst2的概率分布
    logging.info(f"\n计算概率分布（{args.num_bins}个区间）...")
    oasst2_dist, bins = compute_distribution(oasst2_probs, num_bins=args.num_bins)
    magpie_dist_before, _ = compute_distribution(magpie_probs, num_bins=args.num_bins)
    
    logging.info("\noasst2 概率分布:")
    for i in range(len(bins) - 1):
        logging.info(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {oasst2_dist[i]*100:.2f}%")
    
    # 8. 从oasst2随机采样5000条
    logging.info(f"\n从oasst2随机采样 {args.target_samples} 条...")
    oasst2_sampled_items = random.sample([item for item, _ in oasst2_with_probs], 
                                          min(args.target_samples, len(oasst2_with_probs)))
    
    # 9. 根据oasst2分布对magpie进行分层采样
    logging.info(f"\n根据oasst2分布对magpie进行分层采样 {args.target_samples} 条...")
    magpie_sampled_items = stratified_sample_by_distribution(
        magpie_with_probs, 
        oasst2_dist, 
        bins, 
        args.target_samples
    )
    
    # 10. 验证采样后的分布
    magpie_sampled_probs = [prob for item in magpie_sampled_items 
                            for item_data, prob in magpie_with_probs if item_data is item]
    magpie_dist_after, _ = compute_distribution(magpie_sampled_probs, num_bins=args.num_bins)
    
    logging.info("\nmagpie 筛选后概率分布:")
    for i in range(len(bins) - 1):
        logging.info(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {magpie_dist_after[i]*100:.2f}% "
                    f"(目标: {oasst2_dist[i]*100:.2f}%)")
    
    # 11. 计算KL散度
    def kl_divergence(p, q, epsilon=1e-10):
        p = np.array(p) + epsilon
        q = np.array(q) + epsilon
        return np.sum(p * np.log(p / q))
    
    kl_before = kl_divergence(oasst2_dist, magpie_dist_before)
    kl_after = kl_divergence(oasst2_dist, magpie_dist_after)
    
    logging.info(f"\n分布相似度 (KL散度，越小越好):")
    logging.info(f"  筛选前: {kl_before:.6f}")
    logging.info(f"  筛选后: {kl_after:.6f}")
    logging.info(f"  改善程度: {(1 - kl_after/kl_before)*100:.2f}%")
    
    # 12. 合并并保存筛选后的数据
    final_data = oasst2_sampled_items + magpie_sampled_items
    random.shuffle(final_data)
    
    logging.info(f"\n保存筛选后的数据到: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logging.info(f"总数据量: {len(final_data)} (oasst2: {len(oasst2_sampled_items)}, magpie: {len(magpie_sampled_items)})")
    
    # 13. 生成可视化报告
    logging.info("\n生成可视化报告...")
    viz_path = os.path.join(args.output_dir, "distribution_comparison.png")
    visualize_distributions(oasst2_dist, magpie_dist_before, magpie_dist_after, bins, viz_path)
    
    # 14. 保存详细统计报告
    report = {
        "input_file": args.input_file,
        "output_file": args.output_file,
        "total_input": len(all_data),
        "oasst2_input": len(oasst2_data),
        "magpie_input": len(magpie_data),
        "oasst2_sampled": len(oasst2_sampled_items),
        "magpie_sampled": len(magpie_sampled_items),
        "total_output": len(final_data),
        "num_bins": args.num_bins,
        "oasst2_distribution": oasst2_dist.tolist(),
        "magpie_distribution_before": magpie_dist_before.tolist(),
        "magpie_distribution_after": magpie_dist_after.tolist(),
        "bins": bins.tolist(),
        "kl_divergence_before": float(kl_before),
        "kl_divergence_after": float(kl_after),
        "improvement_percent": float((1 - kl_after/kl_before)*100),
        "oasst2_prob_stats": {
            "mean": float(np.mean(oasst2_probs)),
            "std": float(np.std(oasst2_probs)),
            "min": float(np.min(oasst2_probs)),
            "max": float(np.max(oasst2_probs))
        },
        "magpie_prob_stats_before": {
            "mean": float(np.mean(magpie_probs)),
            "std": float(np.std(magpie_probs)),
            "min": float(np.min(magpie_probs)),
            "max": float(np.max(magpie_probs))
        },
        "magpie_prob_stats_after": {
            "mean": float(np.mean(magpie_sampled_probs)),
            "std": float(np.std(magpie_sampled_probs)),
            "min": float(np.min(magpie_sampled_probs)),
            "max": float(np.max(magpie_sampled_probs))
        }
    }
    
    report_path = os.path.join(args.output_dir, "filtering_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logging.info(f"详细报告已保存至: {report_path}")
    
    logging.info("\n" + "=" * 60)
    logging.info("✅ 筛选完成！")
    logging.info("=" * 60)
    logging.info(f"📁 输出文件: {args.output_file}")
    logging.info(f"📊 可视化报告: {viz_path}")
    logging.info(f"📋 详细统计: {report_path}")
    logging.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()
