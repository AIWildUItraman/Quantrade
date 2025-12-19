"""
【生存配置中心】
所有超参数的生死控制台
"""

import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    """数据配置"""
    # 文件路径
    data_path: str = '/home/mengxiaosen/mxs/workspace/Quantrade/data/NEIRO/NERIRO2hLabel.csv'
    
    # 时间列和标签列
    time_col: str = 'time'
    label_col: str = 'label'
    
    # 原始特征列
    price_cols: List[str] = field(default_factory=lambda: ['open', 'high', 'low', 'close'])
    volume_cols: List[str] = field(default_factory=lambda: ['volume', 'amount'])
    
    # 数据集划分比例（严格时间顺序）
    train_ratio: float = 0.95
    val_ratio: float = 0.025
    test_ratio: float = 0.025
    
    # 序列长度（48 = 4天 @ 2H）
    seq_len: int = 32
    
    # 技术指标参数
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    volume_ma_period: int = 20


@dataclass
class ModelConfig:
    """TimesNet模型配置"""
    # 任务类型
    task_name: str = 'classification'
    
    # 输入输出维度
    enc_in: int = 12           # 特征数（动态计算）
    num_class: int = 3         # 类别数：0(无信号), 1(放量上涨), 2(放量下跌)
    c_out: int = 3             # 输出维度
    
    # 序列参数
    seq_len: int = 32
    label_len: int = 0         # 分类任务不需要
    pred_len: int = 0          # 分类任务不需要
    
    # 模型架构
    d_model: int = 64          # 模型维度
    d_ff: int = 128            # FeedForward维度
    e_layers: int = 2          # TimesBlock层数
    
    # TimesNet特有参数
    top_k: int = 3             # FFT提取的主周期数
    num_kernels: int = 6       # Inception卷积核数量
    
    # 正则化
    dropout: float = 0.3
    
    # 时间编码
    embed: str = 'timeF'
    freq: str = 'h'


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    batch_size: int = 32
    val_batch_size: int = 32
    epochs: int = 100
    
    # 优化器
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    
    # 学习率调度
    scheduler_type: str = 'cosine'  # 'cosine' or 'step'
    T_0: int = 10                   # CosineAnnealingWarmRestarts参数
    T_mult: int = 2
    
    # 损失函数（Focal Loss）
    focal_gamma: float = 2.8        # 聚焦参数（越大越关注难样本）
    use_class_weights: bool = True  # 是否使用类别权重
    use_balanced_sampler: bool = True  # 是否按类别均衡采样（提升少数类召回）
    
    # 早停
    patience: int = 19
    min_delta: float = 0.001
    
    # 检查点
    save_dir: str = 'checkpoints'
    save_best_only: bool = True
    
    # 设备
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    
    # 日志
    log_interval: int = 10          # 每N个batch打印一次
    
    # 生存条件阈值
    min_recall_minority: float = 0.30  # 少数类最低召回率
    min_precision_minority: float = 0.40  # 少数类最低精确率


@dataclass
class Config:
    """总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    
    # 随机种子
    seed: int = 42
    
    def __post_init__(self):
        """配置验证"""
        assert self.data.train_ratio + self.data.val_ratio + self.data.test_ratio == 1.0
        assert self.model.seq_len == self.data.seq_len
        
        # 同步序列长度
        self.model.seq_len = self.data.seq_len


# 创建全局配置实例
def get_config():
    """获取配置实例"""
    return Config()
