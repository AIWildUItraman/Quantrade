import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    【特征工程军火库】
    严格因果的技术指标构建
    """
    def __init__(self, config):
        self.config = config.data
        
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建所有特征"""
        df = df.copy()
        
        # === 价格特征 ===
        df['returns'] = df['close'].pct_change(1)
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['close_open_ratio'] = (df['close'] - df['open']) / df['open']
        
        # === 成交量特征（放量检测核心） ===
        vol_ma = df['volume'].rolling(self.config.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / vol_ma
        df['volume_std'] = df['volume'].rolling(self.config.volume_ma_period).std() / vol_ma
        
        amt_ma = df['amount'].rolling(self.config.volume_ma_period).mean()
        df['amount_ratio'] = df['amount'] / amt_ma
        
        # === RSI ===
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(self.config.rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(self.config.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # === MACD ===
        ema_fast = df['close'].ewm(span=self.config.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.config.macd_slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.config.macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # === 布林带 ===
        sma = df['close'].rolling(self.config.bb_period).mean()
        std = df['close'].rolling(self.config.bb_period).std()
        df['bb_upper'] = sma + 2 * std
        df['bb_lower'] = sma - 2 * std
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # === 动量 ===
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        
        # 填充NaN（仅前向填充，保证因果性）
        df = df.fillna(method='ffill').dropna()
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """返回最终特征列"""
        return [
            'close', 'volume', 'returns', 'log_returns',
            'high_low_ratio', 'close_open_ratio',
            'volume_ratio', 'volume_std', 'amount_ratio',
            'rsi', 'macd_hist', 'bb_position', 'momentum_5'
        ]


class CryptoDataset(Dataset):
    """
    【时序数据集】
    严格因果的序列构建
    """
    def __init__(self, df: pd.DataFrame, seq_len: int, feature_cols: List[str], 
                 scaler=None, mode='train'):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.mode = mode
        
        # 因果归一化
        if mode == 'train':
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(df[feature_cols].values)
        else:
            assert scaler is not None, "验证/测试集必须提供训练集的scaler"
            self.scaler = scaler
            self.data = self.scaler.transform(df[feature_cols].values)
        
        self.labels = df['label'].values
        
        # 时间标记（用于mask）
        self.time_marks = np.ones((len(self.data), 1))
    
    def __len__(self):
        # sliding window count; include the last window ending at the final element
        return len(self.data) - self.seq_len + 1
    
    def __getitem__(self, idx):
        # x_enc: [seq_len, features]
        x_enc = self.data[idx:idx + self.seq_len]
        
        # x_mark_enc: [seq_len, 1] 全1表示有效数据
        x_mark_enc = self.time_marks[idx:idx + self.seq_len]
        
        # 标签：序列最后一个时刻的标签
        label = self.labels[idx + self.seq_len - 1]
        
        return {
            'x_enc': torch.FloatTensor(x_enc),
            'x_mark_enc': torch.FloatTensor(x_mark_enc),
            'label': torch.LongTensor([label])
        }


class DataProcessor:
    """
    【数据处理总控】
    """
    def __init__(self, config):
        self.config = config
        self.engineer = FeatureEngineer(config)
        
    def load_and_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载并按时间严格切分数据"""
        # 读取数据
        df = pd.read_csv(self.config.data.data_path)
        df[self.config.data.time_col] = pd.to_datetime(df[self.config.data.time_col])
        df = df.sort_values(self.config.data.time_col).reset_index(drop=True)
        
        # 特征工程
        print("【特征工程】构建技术指标...")
        df = self.engineer.build_features(df)
        
        # 时间切分（绝不shuffle）
        n = len(df)
        train_end = int(n * self.config.data.train_ratio)
        val_end = int(n * (self.config.data.train_ratio + self.config.data.val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        # 验证时间不重叠
        assert train_df[self.config.data.time_col].max() < val_df[self.config.data.time_col].min()
        assert val_df[self.config.data.time_col].max() < test_df[self.config.data.time_col].min()
        
        print(f"【数据切分】")
        print(f"  训练集: {len(train_df)} 样本 ({train_df[self.config.data.time_col].min()} ~ {train_df[self.config.data.time_col].max()})")
        print(f"  验证集: {len(val_df)} 样本 ({val_df[self.config.data.time_col].min()} ~ {val_df[self.config.data.time_col].max()})")
        print(f"  测试集: {len(test_df)} 样本 ({test_df[self.config.data.time_col].min()} ~ {test_df[self.config.data.time_col].max()})")
        
        # 标签分布
        print(f"\n【标签分布】")
        for split_name, split_df in [('训练', train_df), ('验证', val_df), ('测试', test_df)]:
            dist = split_df['label'].value_counts(normalize=True).sort_index()
            print(f"  {split_name}集: ", end='')
            for label, ratio in dist.items():
                print(f"Label {label}: {ratio:.2%}  ", end='')
            print()
        
        return train_df, val_df, test_df
    
    def create_dataloaders(self, train_df, val_df, test_df):
        """创建DataLoader"""
        feature_cols = self.engineer.get_feature_columns()
        seq_len = self.config.data.seq_len
        
        # 更新模型配置中的特征数
        self.config.model.enc_in = len(feature_cols)
        
        # 创建数据集
        train_dataset = CryptoDataset(train_df, seq_len, feature_cols, mode='train')
        val_dataset = CryptoDataset(val_df, seq_len, feature_cols, 
                                    scaler=train_dataset.scaler, mode='val')
        test_dataset = CryptoDataset(test_df, seq_len, feature_cols,
                                     scaler=train_dataset.scaler, mode='test')
        
        # 创建训练采样器：按类别均衡采样，缓解极度不平衡
        if self.config.train.use_balanced_sampler:
            # 对每个可用窗口的标签加权（长度与 __len__ 对齐）
            labels_seq = torch.LongTensor(train_dataset.labels[train_dataset.seq_len - 1:])
            class_sample_count = torch.bincount(labels_seq)
            weight_per_class = 1.0 / torch.clamp(class_sample_count, min=1)
            weights = weight_per_class[labels_seq]
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            shuffle_train = False  # sampler 已经决定顺序
        else:
            sampler = None
            shuffle_train = True

        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=shuffle_train,  # sampler 时不shuffle
            sampler=sampler,
            num_workers=self.config.train.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.train.val_batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.train.val_batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            pin_memory=True
        )
        
        print(f"\n【DataLoader创建完成】")
        print(f"  训练批次数: {len(train_loader)}")
        print(f"  验证批次数: {len(val_loader)}")
        print(f"  测试批次数: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader, train_dataset
