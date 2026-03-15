import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import argparse
import logging
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

# --- 配置区域 (可通过命令行参数覆盖) --- #
DEFAULT_CONFIG = {
    "model_id": "/home/share/models/Llama-3.2-1B",
    "train_file": "./output/train_round1.jsonl",
    "val_file": "./output/val.jsonl",
    "output_dir": "./outputs/experiments/exp001_multiturn_classifier",
    "plot_file": "training_metrics.png",
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "epochs": 3,
    "max_length": 1024,
    "weight_decay": 0.01,
    "seed": 42
}

class TextClassificationDataset(Dataset):
    """用于文本分类的数据集类"""
    def __init__(self, jsonl_file, tokenizer, max_length=1024, preload=True, cache_dir=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.preload = preload
        self.encoded_data = None
        self.cache_dir = cache_dir
        
        # 创建缓存目录
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        # 加载数据
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        # 计算缓存文件名
        if cache_dir:
            # 使用文件路径和最大长度创建唯一的缓存文件名
            import hashlib
            cache_id = hashlib.md5(f"{jsonl_file}_{max_length}_poe".encode()).hexdigest()
            self.cache_file = os.path.join(cache_dir, f"encoded_data_{cache_id}.pt")
        else:
            self.cache_file = None
        
        # 尝试从缓存加载
        if self.cache_file and os.path.exists(self.cache_file):
            logging.info(f"从缓存加载数据: {self.cache_file}")
            try:
                cached_data = torch.load(self.cache_file)

                # 防止缓存与当前数据文件不一致导致 DataLoader 索引越界
                if not isinstance(cached_data, list):
                    raise ValueError(f"缓存格式错误，期望 list，实际为 {type(cached_data)}")
                if len(cached_data) != len(self.data):
                    raise ValueError(
                        f"缓存条数({len(cached_data)})与当前数据条数({len(self.data)})不一致"
                    )

                self.encoded_data = cached_data
                logging.info(f"成功从缓存加载 {len(self.encoded_data)} 条数据")
                return
            except Exception as e:
                logging.warning(f"加载缓存失败或缓存失效: {e}，将重新处理数据")
        
        # 预处理并缓存
        if preload:
            logging.info(f"预加载并编码数据集: {jsonl_file}")
            self.encoded_data = []
            for item in tqdm(self.data, desc="加载数据"):
                encoded = self._encode_item(item)
                self.encoded_data.append(encoded)
            
            # 保存到缓存
            if self.cache_file:
                logging.info(f"保存编码数据到缓存: {self.cache_file}")
                torch.save(self.encoded_data, self.cache_file)
    
    def _encode_item(self, item):
        """将单个样本编码为模型输入"""
        # 支持三种数据格式：
        # 1. 包含 "conversations" 字段的格式（本项目多轮数据）
        #    每个 turn: {"role": "user"/"assistant", "content": "..."}
        #    label: 0=oasst2, 1=Magpie
        # 2. 包含 "text" 字段的格式
        # 3. 包含 "instruction" 和 "response" 字段的格式
        if "conversations" in item:
            # 将多轮对话拼接为纯文本，格式：
            # User: ...\n\nAssistant: ...\n\nUser: ...
            parts = []
            for turn in item["conversations"]:
                role = "User" if turn["role"] == "user" else "Assistant"
                parts.append(f"{role}: {turn['content']}")
            text = "\n\n".join(parts)
        elif "text" in item:
            text = item["text"]
        else:
            # 从 instruction 和 response 构建 text
            text = f"Question: {item['instruction']}\nAnswer: {item['response']}"

        label = item["label"]
        
        # 使用tokenizer编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 压缩第一个维度
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # 计算实际长度（非padding token的数量）
        seq_length = attention_mask.sum().item()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
            "seq_length": torch.tensor(seq_length, dtype=torch.float32)
        }

    def __len__(self):
        # 预编码场景下与 encoded_data 对齐，避免出现长度不一致
        if self.encoded_data is not None:
            return len(self.encoded_data)
        return len(self.data)

    def __getitem__(self, idx):
        if self.encoded_data is not None:
            return self.encoded_data[idx]
        
        item = self.data[idx]
        return self._encode_item(item)


class LengthClassifier(nn.Module):
    """
    简单的长度分类器 - 仅基于序列长度进行分类
    这是一个极其简单的两层MLP，输入是序列长度，输出是类别logits
    """
    def __init__(self, num_labels=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels)
        )
    
    def forward(self, seq_lengths):
        """
        Args:
            seq_lengths: (batch_size,) tensor of sequence lengths
        Returns:
            logits: (batch_size, num_labels) tensor
        """
        # 将长度reshape为(batch_size, 1)
        x = seq_lengths.unsqueeze(-1)
        logits = self.net(x)
        return logits


class ClassifierWithLoRAAndPoE(nn.Module):
    """
    使用LoRA的分类器模型，带有自定义分类头和Product of Experts机制
    
    PoE (Product of Experts) 机制：
    - 主模型（基于内容的分类器）: 提取语义特征进行分类
    - 长度专家（基于长度的分类器）: 仅基于序列长度进行分类
    - 最终logits = 主模型logits + 长度专家logits
    
    工作原理：
    1. 当数据符合长度刻板印象时（如长文本=Magpie），长度专家会给出正确预测，
       主模型无需学习，梯度很小
    2. 当数据违反长度刻板印象时（如短文本=Magpie），长度专家预测错误，
       产生大的loss，迫使主模型学习真正的语义特征
    """
    def __init__(self, model_id, num_labels=2, use_lora=True, tokenizer=None, use_poe=True):
        super().__init__()
        
        # 加载配置
        self.config = AutoConfig.from_pretrained(model_id)
        self.num_labels = num_labels
        self.use_poe = use_poe
        
        # 确保配置中有padding_token_id
        if tokenizer is not None:
            self.config.pad_token_id = tokenizer.pad_token_id
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=self.config,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # 确保模型知道padding token
        if tokenizer is not None:
            self.model.config.pad_token_id = tokenizer.pad_token_id
        
        # 应用LoRA
        if use_lora:
            logging.info("应用LoRA配置到模型...")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # 获取模型设备
        device = next(self.model.parameters()).device
        
        # 添加主分类头（基于内容的分类器）
        hidden_size = self.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels, dtype=torch.bfloat16).to(device)
        
        # 添加长度分类器（Product of Experts中的"长度专家"）
        if self.use_poe:
            self.length_classifier = LengthClassifier(num_labels=num_labels).to(device)
            logging.info("✅ 已启用 Product of Experts (PoE) 机制")
            logging.info("   - 主模型: 基于语义内容的分类器")
            logging.info("   - 长度专家: 仅基于序列长度的简单分类器")
            logging.info("   - 最终预测: logits_total = logits_main + logits_length")
        else:
            self.length_classifier = None
            logging.info("未启用 PoE 机制，使用标准分类器")
        
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss().to(device)
        
        logging.info(f"模型设备: {device}, 分类器设备: {next(self.classifier.parameters()).device}")
    
    def forward(self, input_ids, attention_mask, seq_lengths=None, labels=None):
        """
        前向传播
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            seq_lengths: (batch_size,) - 序列的实际长度（用于长度分类器）
            labels: (batch_size,) - 真实标签
        """
        # 获取模型输出 - 只保留最后一层的隐藏状态以节省内存
        with torch.cuda.amp.autocast(enabled=True):  # 使用混合精度训练
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # 获取最后一层的隐藏状态
            hidden_states = outputs.hidden_states[-1]
            
            # 立即删除不需要的隐藏状态以释放内存
            del outputs.hidden_states
        
        # 获取每个序列的最后一个非填充token的隐藏状态作为特征
        # 计算每个序列的实际长度
        seq_lengths_from_mask = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        
        # 确保索引不会越界
        seq_lengths_from_mask = torch.clamp(seq_lengths_from_mask, max=hidden_states.shape[1]-1)
        
        # 收集每个序列的最后一个token的隐藏状态
        pooled_output = torch.stack([hidden_states[i, seq_lengths_from_mask[i], :] for i in range(batch_size)])
        
        # 确保pooled_output在正确的设备上和正确的数据类型
        device = next(self.classifier.parameters()).device
        dtype = next(self.classifier.parameters()).dtype
        pooled_output = pooled_output.to(device=device, dtype=dtype)
        
        # 通过主分类头获取logits（基于语义内容）
        logits_main = self.classifier(pooled_output)
        
        # Product of Experts: 结合长度专家的预测
        if self.use_poe and seq_lengths is not None:
            # 确保seq_lengths在正确的设备上
            seq_lengths = seq_lengths.to(device)
            
            # 长度分类器的logits（仅基于序列长度）
            logits_length = self.length_classifier(seq_lengths)
            
            # PoE机制: 简单相加（在log空间相当于概率相乘）
            logits = logits_main + logits_length
        else:
            logits = logits_main
        
        # 计算损失
        loss = None
        if labels is not None:
            # 确保labels在正确的设备上
            labels = labels.to(device)
            loss = self.loss_fn(logits, labels)
        
        # 返回与transformers模型输出格式兼容的结果
        from types import SimpleNamespace
        return SimpleNamespace(
            loss=loss, 
            logits=logits,
            logits_main=logits_main,  # 返回主模型的logits用于分析
            logits_length=logits_length if self.use_poe and seq_lengths is not None else None
        )


def evaluate(model, dataloader, device, label_names=None):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    num_batches = len(dataloader)
    if num_batches == 0:
        raise ValueError("验证DataLoader为空（0个batch）。请检查验证集文件是否为空、路径是否正确。")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            seq_lengths = batch["seq_length"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                seq_lengths=seq_lengths,
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算平均损失
    avg_loss = total_loss / num_batches
    
    # 生成分类报告
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=label_names, 
        digits=4, 
        zero_division=0
    )
    
    # 计算准确率、精确率、召回率和F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics, report, all_labels, all_preds


def collate_fn(batch):
    """将批次数据整合为模型输入格式"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    seq_lengths = torch.stack([item["seq_length"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels,
        "seq_length": seq_lengths
    }


def save_text(path, content):
    """保存文本到文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def save_confusion_matrix_figure(labels, preds, label_names, save_path, title="混淆矩阵"):
    """保存混淆矩阵图"""
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = range(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45)
    plt.yticks(tick_marks, label_names)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("真实标签")
    plt.xlabel("预测标签")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def train():
    """主训练函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="训练带有Product of Experts机制的二分类器模型")
    parser.add_argument("--model_id", type=str, default=DEFAULT_CONFIG["model_id"], help="模型ID或路径")
    parser.add_argument("--train_file", type=str, default=DEFAULT_CONFIG["train_file"], help="训练数据文件路径")
    parser.add_argument("--val_file", type=str, default=DEFAULT_CONFIG["val_file"], help="验证数据文件路径")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"], help="输出目录")
    parser.add_argument("--plot_file", type=str, default=DEFAULT_CONFIG["plot_file"], help="损失曲线图文件名")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"], help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_CONFIG["gradient_accumulation_steps"], help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"], help="学习率")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"], help="训练轮数")
    parser.add_argument("--max_length", type=int, default=DEFAULT_CONFIG["max_length"], help="最大序列长度")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"], help="权重衰减")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"], help="随机种子")
    parser.add_argument("--no_lora", action="store_true", help="不使用LoRA")
    parser.add_argument("--no_preload", action="store_true", help="不预加载数据到内存")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="数据缓存目录")
    parser.add_argument("--no_cache", action="store_true", help="不使用数据缓存")
    parser.add_argument("--batch_size_1", action="store_true", help="强制使用批次大小为1（解决某些模型的padding问题）")
    parser.add_argument("--save_steps", type=int, default=1000, help="每多少步保存一次检查点")
    parser.add_argument("--gc_steps", type=int, default=50, help="每多少步执行一次垃圾回收")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪的最大范数")
    
    # PoE相关参数
    parser.add_argument("--no_poe", action="store_true", help="禁用Product of Experts机制")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    reports_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # 保存训练配置
    config_dict = vars(args)
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    logging.info(f"训练配置已保存至: {config_path}")
    
    # 1. 加载 Tokenizer
    logging.info(f"正在加载 Tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 设置右填充（分类任务必须使用right padding，配合last token pooling）
    tokenizer.padding_side = "right"
    
    # 2. 加载数据集
    logging.info("正在加载数据集...")
    cache_dir = None if args.no_cache else args.cache_dir
    
    train_dataset = TextClassificationDataset(
        args.train_file, 
        tokenizer, 
        max_length=args.max_length, 
        preload=not args.no_preload,
        cache_dir=cache_dir
    )
    val_dataset = TextClassificationDataset(
        args.val_file, 
        tokenizer, 
        max_length=args.max_length, 
        preload=not args.no_preload,
        cache_dir=cache_dir
    )

    # 训练前数据完整性检查
    if len(train_dataset) == 0:
        raise ValueError(f"训练集为空: {args.train_file}。请检查文件内容与数据预处理流程。")
    if len(val_dataset) == 0:
        raise ValueError(f"验证集为空: {args.val_file}。请检查文件内容与数据预处理流程。")

    # 创建数据加载器
    # 如果指定了batch_size_1参数，则强制使用批次大小为1
    effective_batch_size = 1 if args.batch_size_1 else args.batch_size
    
    # 减少worker数量和预取因子以降低内存使用
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,  # 减少worker数量
        pin_memory=True,
        prefetch_factor=2,  # 减少预取因子
        persistent_workers=False  # 不使用持久化worker
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,  # 减少worker数量
        pin_memory=True,
        prefetch_factor=2  # 减少预取因子
    )
    
    # 3. 初始化模型
    logging.info(f"正在初始化模型: {args.model_id}")
    model = ClassifierWithLoRAAndPoE(
        args.model_id,
        num_labels=2,
        use_lora=not args.no_lora,
        tokenizer=tokenizer,
        use_poe=not args.no_poe
    )
    
    # 4. 设置优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 5. 检查是否有检查点可以恢复
    start_epoch = 0
    best_val_accuracy = 0.0
    checkpoints = []
    
    if os.path.isdir(args.output_dir):
        checkpoints = [f for f in os.listdir(args.output_dir) 
                      if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
    
    if checkpoints:
        # 按照轮数排序
        checkpoints.sort(key=lambda x: int(re.search(r"epoch_(\d+)", x).group(1)), reverse=True)
        latest_ckpt = os.path.join(args.output_dir, checkpoints[0])
        logging.info(f"找到最新检查点: {latest_ckpt}，正在加载...")
        
        try:
            checkpoint = torch.load(latest_ckpt, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_accuracy = checkpoint.get("best_val_accuracy", 0.0)
            
            logging.info(f"成功从轮次 {start_epoch-1} 恢复。从轮次 {start_epoch} 开始训练。")
        except Exception as e:
            logging.error(f"加载检查点失败: {e}。从头开始训练。")
    
    # 6. 训练循环
    device = next(model.parameters()).device
    logging.info(f"使用设备: {device}")
    
    # 用于绘图的数据
    train_losses = []
    train_steps = []
    val_metrics = []
    val_epochs = []
    
    # 标签名称
    label_names = ["oasst2 (human)", "Magpie (synthetic)"]
    
    metrics_history = []

    # 开始训练
    logging.info("🚀 开始训练...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        step = 0
        
        # 训练一个轮次
        pbar = tqdm(train_loader, desc=f"轮次 {epoch+1}/{args.epochs}")
        for i, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            seq_lengths = batch["seq_length"].to(device)
            labels = batch["label"].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                seq_lengths=seq_lengths,
                labels=labels
            )
            loss = outputs.loss
            
            # 梯度累积
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            # 更新参数
            if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                
                # 定期执行垃圾回收以释放内存
                if step % args.gc_steps == 0:
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                
                # 定期保存检查点，避免长时间训练后内存溢出导致全部丢失
                if step % args.save_steps == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_step_{step}_epoch_{epoch+1}.pt")
                    torch.save({
                        "epoch": epoch,
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": loss.item() * args.gradient_accumulation_steps
                    }, checkpoint_path)
                    logging.info(f"保存中间检查点: {checkpoint_path}")
                
                # 记录损失
                epoch_loss += loss.item() * args.gradient_accumulation_steps
                train_losses.append(loss.item() * args.gradient_accumulation_steps)
                train_steps.append(epoch * len(train_loader) + i)
                
                # 更新进度条
                gpu_mem = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
                pbar.set_postfix({
                    "loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                    "gpu_mem": f"{gpu_mem:.2f}GB"
                })
        
        # 计算平均损失
        avg_train_loss = epoch_loss / (len(train_loader) / args.gradient_accumulation_steps)
        
        # 评估模型
        logging.info(f"评估轮次 {epoch+1}...")
        val_metrics_dict, val_report, val_labels, val_preds = evaluate(
            model, val_loader, device, label_names=label_names
        )
        
        # 记录验证指标
        val_metrics.append(val_metrics_dict)
        val_epochs.append(epoch + 1)
        
        # 打印结果
        logging.info(f"\n轮次 {epoch+1} 结果:")
        logging.info(f"训练损失: {avg_train_loss:.4f}")
        logging.info(f"验证损失: {val_metrics_dict['loss']:.4f}, 验证准确率: {val_metrics_dict['accuracy']:.4f}")
        logging.info(f"验证精确率: {val_metrics_dict['precision']:.4f}, 验证召回率: {val_metrics_dict['recall']:.4f}, 验证F1: {val_metrics_dict['f1']:.4f}")
        logging.info("详细分类报告:")
        logging.info(val_report)

        # 保存每个epoch的分类报告
        epoch_report_path = os.path.join(reports_dir, f"epoch_{epoch+1}_classification_report.txt")
        save_text(epoch_report_path, val_report)

        # 记录指标历史
        metrics_history.append({
            "epoch": epoch + 1,
            "train_loss": float(avg_train_loss),
            "val_loss": float(val_metrics_dict['loss']),
            "accuracy": float(val_metrics_dict['accuracy']),
            "precision": float(val_metrics_dict['precision']),
            "recall": float(val_metrics_dict['recall']),
            "f1": float(val_metrics_dict['f1'])
        })
        
        # 保存检查点
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}_acc_{val_metrics_dict['accuracy']:.4f}.pt")
        
        # 保存当前模型状态
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_metrics": val_metrics_dict,
            "best_val_accuracy": max(best_val_accuracy, val_metrics_dict['accuracy'])
        }, checkpoint_path)
        
        # 如果是最佳模型，保存为best_model
        if val_metrics_dict['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics_dict['accuracy']
            best_model_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_metrics": val_metrics_dict,
                "best_val_accuracy": best_val_accuracy
            }, best_model_path)

            # 同步保存最佳分类报告和混淆矩阵
            best_report_path = os.path.join(reports_dir, "best_classification_report.txt")
            save_text(best_report_path, val_report)
            save_confusion_matrix_figure(
                val_labels,
                val_preds,
                label_names,
                os.path.join(args.output_dir, "confusion_matrix_best.png"),
                title=f"最佳模型混淆矩阵 (Epoch {epoch+1})"
            )

            logging.info(f"✨ 保存最佳模型，验证准确率: {best_val_accuracy:.4f}")
    
    # 保存指标历史
    metrics_history_path = os.path.join(args.output_dir, "metrics_history.json")
    with open(metrics_history_path, "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, ensure_ascii=False, indent=2)

    # 7. 绘制训练曲线
    if train_losses:
        logging.info(f"正在生成训练曲线图: {args.plot_file}")
        plt.figure(figsize=(12, 8))
        
        # 绘制训练损失
        plt.subplot(2, 1, 1)
        plt.plot(train_steps, train_losses, label="训练损失")
        plt.xlabel("步数")
        plt.ylabel("损失")
        plt.title("训练损失曲线")
        plt.legend()
        
        # 绘制验证指标
        plt.subplot(2, 1, 2)
        epochs_range = val_epochs
        plt.plot(epochs_range, [m['accuracy'] for m in val_metrics], 'o-', label="准确率")
        plt.plot(epochs_range, [m['precision'] for m in val_metrics], 's-', label="精确率")
        plt.plot(epochs_range, [m['recall'] for m in val_metrics], '^-', label="召回率")
        plt.plot(epochs_range, [m['f1'] for m in val_metrics], 'd-', label="F1分数")
        plt.xlabel("轮次")
        plt.ylabel("分数")
        plt.title("验证指标")
        plt.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, args.plot_file)
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"训练曲线已保存至: {plot_path}")
    
    # 8. 最终评估并导出分类报告
    logging.info("执行最终评估并导出报告...")
    final_metrics, final_report, final_labels, final_preds = evaluate(
        model, val_loader, device, label_names=label_names
    )

    save_text(os.path.join(reports_dir, "final_classification_report.txt"), final_report)
    save_confusion_matrix_figure(
        final_labels,
        final_preds,
        label_names,
        os.path.join(args.output_dir, "confusion_matrix_final.png"),
        title="最终模型混淆矩阵"
    )

    with open(os.path.join(args.output_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in final_metrics.items()}, f, ensure_ascii=False, indent=2)

    # 9. 保存最终模型
    logging.info("保存最终模型...")
    
    # 保存完整模型
    final_model_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    model.model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # 如果使用了LoRA，保存LoRA适配器
    if not args.no_lora:
        lora_dir = os.path.join(args.output_dir, "lora_adapter")
        os.makedirs(lora_dir, exist_ok=True)
        model.model.save_pretrained(lora_dir)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"✅ 训练完成！")
    logging.info(f"{'='*60}")
    logging.info(f"📂 模型保存位置: {args.output_dir}")
    logging.info(f"   - 完整模型: {final_model_dir}")
    if not args.no_lora:
        logging.info(f"   - LoRA适配器: {lora_dir}")
    logging.info(f"🎯 最佳验证准确率: {best_val_accuracy:.4f}")
    if not args.no_poe:
        logging.info(f"✨ Product of Experts 机制已启用，模型已学会忽略长度偏见")
    logging.info(f"{'='*60}\n")

if __name__ == "__main__":
    train()
