# NEIRO 虚拟货币 2 小时 K 线基于 TimesNet 的完整训练方案

## ⚠️ 核心原则：零未来函数容忍（Zero Look-ahead Tolerance）

> **本方案的最高优先级：确保模型在时刻 T 的预测仅依赖于 T 及之前的信息，严格遵循因果性（Causality）原则。**

---

## 1. 防未来函数的数据预处理 (Anti-Leakage Preprocessing)

### 1.1 时序划分 (Temporal Splitting)

```python
import pandas as pd

# 读取数据并按时间排序
df = pd.read_csv('neiro_2h.csv')
df = df.sort_values('time').reset_index(drop=True)  # 严格时序排序

# 计算划分点
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df[:train_end]
val_df = df[train_end:val_end]
test_df = df[val_end:]
```

**🚨 严禁操作：**
- ❌ 使用 `train_test_split` 的 `shuffle=True` 参数
- ❌ 随机抽样验证集
- ❌ K-Fold 交叉验证（会破坏时间顺序）

**✅ 正确理念：**
- 训练集必须在时间轴上先于验证集
- 验证集必须在时间轴上先于测试集
- 模拟真实交易环境：模型在 T 时刻只能看到历史数据

---

### 1.2 因果归一化 (Causal Normalization)

#### ❌ 错误做法（未来函数陷阱）

```python
# 这会导致数据泄露！
scaler = MinMaxScaler()
df[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(
    df[['open', 'high', 'low', 'close', 'volume']]
)
```

**问题根源：**
- `fit_transform` 使用了整个数据集的统计量（包括未来的 max/min）
- 模型在训练时已"见过"测试集的分布特征
- 等同于"穿越回过去用未来信息下注"

---

#### ✅ 方案 A: 固定训练集统计量归一化

```python
from sklearn.preprocessing import StandardScaler

# 1. 仅在训练集上拟合 scaler
feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
scaler = StandardScaler()
scaler.fit(train_df[feature_cols])

# 2. 分别转换三个集合
train_df[feature_cols] = scaler.transform(train_df[feature_cols])
val_df[feature_cols] = scaler.transform(val_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])
```

**适用场景：**
- 市场环境相对稳定
- 训练集时间跨度足够长（至少覆盖 1-2 个完整市场周期）

---

#### ✅ 方案 B: 滚动窗口归一化（更严格）

```python
def rolling_normalize(df, feature_cols, window=168):  # 168 = 2周的2小时K线
    """
    使用滚动窗口进行因果归一化
    对于时刻 t，仅使用 [t-window, t-1] 的统计量
    """
    normalized_df = df.copy()
  
    for col in feature_cols:
        means = df[col].rolling(window=window, min_periods=1).mean().shift(1)
        stds = df[col].rolling(window=window, min_periods=1).std().shift(1)
        stds = stds.replace(0, 1)  # 避免除零
      
        normalized_df[col] = (df[col] - means) / stds
  
    return normalized_df

# 分别对三个集合进行滚动归一化
train_df = rolling_normalize(train_df, feature_cols, window=168)
val_df = rolling_normalize(val_df, feature_cols, window=168)
test_df = rolling_normalize(test_df, feature_cols, window=168)
```

**优势：**
- 完全避免使用未来信息
- 适应市场环境变化（非平稳性）
- 模拟实盘环境（每个时刻仅知道历史统计）

**注意 `shift(1)` 的关键作用：**
- 确保时刻 t 的归一化参数来自 [t-window, t-1]
- 避免使用当前时刻自身的信息

---

### 1.3 特征审查 (Feature Causality Check)

在构建技术指标前，必须遵循以下检查清单：

| 操作类型 | 是否允许 | 示例 |
|---------|---------|------|
| 滞后操作 `shift(n)`, n>0 | ✅ | `df['close'].shift(1)` |
| 向前操作 `shift(-n)` | ❌ | `df['close'].shift(-1)` (泄露未来) |
| 累积操作 (expanding) | ✅ | `df['volume'].expanding().sum()` |
| 中心化滑窗 `rolling(center=True)` | ❌ | 使用了前后对称的数据 |
| 因果滑窗 `rolling(center=False)` | ✅ | 默认行为，仅使用历史数据 |

---

## 2. 特征工程 (Feature Engineering)

### 2.1 基础特征

```python
base_features = ['open', 'high', 'low', 'close', 'volume', 'amount']
```

### 2.2 衍生技术指标（严格因果设计）

#### 指标 1: 相对强弱指数 (RSI)

```python
def causal_rsi(close, period=14):
    """
    因果 RSI 计算
    在时刻 t，仅使用 [t-period, t] 的价格数据
    """
    delta = close.diff()  # 默认 shift(1) 行为
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
  
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['rsi'] = causal_rsi(df['close'], period=14)
```

**因果性保证：**
- `diff()` 默认计算 `close[t] - close[t-1]`
- `rolling()` 默认 `center=False`，只向后看

---

#### 指标 2: 能量潮 (OBV - On-Balance Volume)

```python
def causal_obv(close, volume):
    """
    OBV 天然是因果指标
    OBV[t] = OBV[t-1] + sign(close[t] - close[t-1]) * volume[t]
    """
    obv = (volume * (~close.diff().le(0) * 2 - 1)).cumsum()
    return obv

df['obv'] = causal_obv(df['close'], df['volume'])
```

**用途：**
- 捕捉"放量"特征
- 特别适合本任务的 Label 1（放量上涨）和 Label 2（放量下跌）

---

#### 指标 3: 平均真实波幅 (ATR - Average True Range)

```python
def causal_atr(high, low, close, period=14):
    """
    ATR 衡量市场波动性
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
  
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

df['atr'] = causal_atr(df['high'], df['low'], df['close'], period=14)
```

---

#### 指标 4: MACD (移动平均收敛/发散)

```python
def causal_macd(close, fast=12, slow=26, signal=9):
    """
    MACD 基于指数移动平均（EMA）
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
  
    return macd, signal_line, histogram

df['macd'], df['macd_signal'], df['macd_hist'] = causal_macd(df['close'])
```

**EMA 的因果性：**
- `ewm(adjust=False)` 确保递归计算：`EMA[t] = α * Price[t] + (1-α) * EMA[t-1]`

---

#### 指标 5: 价格变化率 (Rate of Change)

```python
def causal_roc(close, period=12):
    """
    ROC[t] = (close[t] - close[t-period]) / close[t-period]
    """
    roc = close.pct_change(periods=period) * 100
    return roc

df['roc'] = causal_roc(df['close'], period=12)
```

---

### 2.3 滑动窗口序列构建 (Sliding Window)

```python
import numpy as np

def create_sequences(df, feature_cols, seq_len=96):
    """
    为时刻 T 构建输入序列 [T-seq_len+1, ..., T]
  
    Args:
        df: 已排序的 DataFrame
        feature_cols: 特征列名列表
        seq_len: 序列长度（96 对应 2小时 * 96 = 8 天历史）
  
    Returns:
        X: (N, seq_len, n_features)
        y: (N,)
    """
    X, y = [], []
    data = df[feature_cols].values
    labels = df['label'].values
  
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])  # [i-96, ..., i-1, i] 共96个点
        y.append(labels[i])
  
    return np.array(X), np.array(y)

# 特征列表（包含基础特征 + 技术指标）
feature_list = [
    'open', 'high', 'low', 'close', 'volume', 'amount',
    'rsi', 'obv', 'atr', 'macd', 'macd_signal', 'macd_hist', 'roc'
]

X_train, y_train = create_sequences(train_df, feature_list, seq_len=96)
X_val, y_val = create_sequences(val_df, feature_list, seq_len=96)
X_test, y_test = create_sequences(test_df, feature_list, seq_len=96)

print(f"训练集形状: {X_train.shape}, 标签形状: {y_train.shape}")
# 输出示例: (10000, 96, 13), (10000,)
```

**关键点：**
- 序列长度 96 = 8 天的 2 小时 K 线
- 对于时刻 i，输入窗口是 `[i-95, ..., i-1, i]`
- 标签是时刻 i 的 label（当前时刻的信号类型）

---

## 3. 解决类别极度不平衡 (Handling Imbalance)

### 3.1 样本分布分析

```python
from collections import Counter

print("训练集标签分布:")
print(Counter(y_train))
# 示例输出: {0: 8900, 1: 600, 2: 500}

# 计算类别权重
class_counts = np.bincount(y_train)
class_weights = len(y_train) / (len(class_counts) * class_counts)
print("类别权重:", class_weights)
# 示例输出: [0.37, 5.56, 6.67]
```

---

### 3.2 方案 A: 加权交叉熵损失

```python
import torch
import torch.nn as nn

class_weights_tensor = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
```

**原理：**
- 对稀有类别（1 和 2）赋予更高的损失权重
- 迫使模型关注少数类样本

---

### 3.3 方案 B: Focal Loss（推荐）

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, reduction='mean'):
        """
        Focal Loss for imbalanced classification
      
        Args:
            alpha: 类别权重 (可选)
            gamma: 聚焦参数，gamma > 1 降低易分类样本的损失
            reduction: 'mean' 或 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
  
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
      
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 使用建议
alpha = torch.FloatTensor([1.0, 5.0, 5.0]).to(device)  # 手动调整权重
criterion = FocalLoss(alpha=alpha, gamma=2.5)
```

**参数调优建议：**
- `gamma=2.0`: 轻度聚焦
- `gamma=2.5`: 中度聚焦（推荐起点）
- `gamma=3.0`: 强度聚焦（适合极度不平衡）

**为什么 Focal Loss 适合本任务：**
- 89% 的 Label 0 样本很容易被分类，Focal Loss 会自动降低它们的损失贡献
- 迫使模型将计算资源集中在难以分类的 Label 1 和 2

---

### 3.4 过采样的严格约束

**🚨 如果使用 SMOTE 等过采样：**

```python
from imblearn.over_sampling import SMOTE

# ✅ 正确做法：仅对训练集进行过采样
X_train_flat = X_train.reshape(X_train.shape[0], -1)
smote = SMOTE(sampling_strategy={1: 3000, 2: 3000}, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train)
X_train_resampled = X_train_resampled.reshape(-1, 96, len(feature_list))

# ❌ 严禁操作：在时序数据上进行交叉采样会打破时间结构
```

**替代建议：**
- 优先使用 Focal Loss 或类别权重
- 如必须过采样，考虑仅在特定时间段内进行（避免跨周期泄露）

---

## 4. TimesNet 模型架构配置

### 4.1 TimesNet 核心原理

TimesNet 将一维时间序列转换为二维张量，以捕捉多周期性模式：

1. **周期检测**：通过 FFT（快速傅里叶变换）识别主要周期
2. **2D 重塑**：将序列 reshape 为 `(period, seq_len // period)` 的 2D 图像
3. **Inception Block**：用多尺度卷积捕捉局部和全局模式
4. **自适应聚合**：融合不同周期的特征

**适用于 2 小时 K 线的原因：**
- 虚拟货币市场存在日内周期（12 小时 = 6 个 2h K 线）
- 周末效应、资金费率周期（8 小时 = 4 个 2h K 线）

---

### 4.2 模型配置

```python
class TimesNetConfig:
    # 数据参数
    seq_len = 96          # 输入序列长度（8天历史）
    pred_len = 0          # 预测长度（分类任务设为0）
    label_len = 0         # 仅用于预测任务
  
    # 特征维度
    enc_in = 13           # 输入特征数（6个基础 + 7个技术指标）
    c_out = 3             # 输出类别数（0, 1, 2）
  
    # TimesNet 核心参数
    d_model = 64          # 模型隐藏维度
    d_ff = 128            # FeedForward 维度
    num_kernels = 6       # Inception 中的卷积核数量
    top_k = 5             # 选择 Top-K 个周期
  
    # 模型结构
    e_layers = 2          # Encoder 层数
    dropout = 0.1
  
    # 训练参数
    batch_size = 64
    learning_rate = 1e-4
    epochs = 100
    patience = 15         # 早停耐心值

config = TimesNetConfig()
```

---

### 4.3 参数选择依据

| 参数 | 推荐值 | 理由 |
|------|--------|------|
| `seq_len=96` | 8 天 | 覆盖 1 周以上历史，捕捉周末效应 |
| `top_k=5` | 5 个周期 | 捕捉多时间尺度（4h, 8h, 12h, 24h, 48h） |
| `d_model=64` | 64 | 平衡表达能力与过拟合风险 |
| `e_layers=2` | 2 层 | 更多层可能导致过拟合（数据量有限） |

---

### 4.4 模型实例化（伪代码）

```python
from timesnet import TimesNet  # 假设已安装

model = TimesNet(
    seq_len=config.seq_len,
    pred_len=config.pred_len,
    enc_in=config.enc_in,
    c_out=config.c_out,
    d_model=config.d_model,
    d_ff=config.d_ff,
    num_kernels=config.num_kernels,
    top_k=config.top_k,
    e_layers=config.e_layers,
    dropout=config.dropout,
    task_name='classification'  # 关键设置
)

model = model.to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 5. 训练与评估 (Training & Evaluation)

### 5.1 数据加载器

```python
from torch.utils.data import TensorDataset, DataLoader

# 转换为 PyTorch 张量
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.LongTensor(y_val)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 训练时可 shuffle
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)     # 验证时禁止 shuffle
```

**注意：**
- 训练时可以 shuffle（因为已通过滑窗构建独立样本）
- 但原始时序数据划分时必须严格按时间顺序

---

### 5.2 优化器配置

```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5  # L2 正则化
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',          # 监控指标越大越好
    factor=0.5,          # 学习率衰减因子
    patience=5,
    verbose=True
)
```

---

### 5.3 训练循环（含早停）

```python
from sklearn.metrics import f1_score, classification_report, confusion_matrix

best_val_f1 = 0.0
patience_counter = 0

for epoch in range(config.epochs):
    # ============ 训练阶段 ============
    model.train()
    train_loss = 0.0
  
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
      
        optimizer.zero_grad()
        outputs = model(batch_x)  # (batch, 3)
        loss = criterion(outputs, batch_y)
      
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
      
        train_loss += loss.item()
  
    avg_train_loss = train_loss / len(train_loader)
  
    # ============ 验证阶段 ============
    model.eval()
    val_preds, val_labels = [], []
  
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
          
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
          
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(batch_y.cpu().numpy())
  
    # 计算 Macro F1（关键指标）
    val_f1 = f1_score(val_labels, val_preds, average='macro')
  
    print(f"Epoch {epoch+1}/{config.epochs}")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Macro F1: {val_f1:.4f}")
  
    # 学习率调度
    scheduler.step(val_f1)
  
    # ============ 早停逻辑 ============
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print("  ✅ 模型已保存")
    else:
        patience_counter += 1
        if patience_counter >= config.patience:
            print(f"  ⛔ 早停触发（连续 {config.patience} 轮无改善）")
            break
```

---

### 5.4 测试集评估

```python
# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.LongTensor(y_test).to(device)

with torch.no_grad():
    test_outputs = model(X_test_t)
    test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()
    test_labels = y_test_t.cpu().numpy()

# ============ 详细评估报告 ============
print("=" * 60)
print("测试集评估报告")
print("=" * 60)

# 1. 混淆矩阵
cm = confusion_matrix(test_labels, test_preds)
print("\n混淆矩阵:")
print("              预测 0    预测 1    预测 2")
for i, row in enumerate(cm):
    print(f"  实际 {i}      {row[0]:6}    {row[1]:6}    {row[2]:6}")

# 2. 分类报告（含 Precision, Recall, F1）
print("\n分类报告:")
print(classification_report(
    test_labels, 
    test_preds, 
    target_names=['无信号(0)', '放量上涨(1)', '放量下跌(2)'],
    digits=4
))

# 3. 各类别 Recall（重点关注）
from sklearn.metrics import recall_score
recall_per_class = recall_score(test_labels, test_preds, average=None)
print("\n各类别召回率（Recall）:")
print(f"  Label 0 (无信号):   {recall_per_class[0]:.4f}")
print(f"  Label 1 (放量上涨): {recall_per_class[1]:.4f}  ← 关键指标")
print(f"  Label 2 (放量下跌): {recall_per_class[2]:.4f}  ← 关键指标")

# 4. Macro F1
macro_f1 = f1_score(test_labels, test_preds, average='macro')
print(f"\nMacro F1-score: {macro_f1:.4f}")
```

---

### 5.5 评估指标解读

**为什么不能只看 Accuracy？**

假设测试集有 1000 个样本：
- Label 0: 890 个
- Label 1: 60 个
- Label 2: 50 个

如果模型将所有样本预测为 0，Accuracy = 89%，但完全没有捕捉到交易信号！

**正确的评估体系：**

| 指标 | 含义 | 本任务的重点 |
|------|------|-------------|
| **Precision (查准率)** | 预测为正类中真正为正类的比例 | 避免虚假信号（降低交易成本） |
| **Recall (查全率)** | 实际正类中被预测出的比例 | **捕捉放量机会（核心目标）** |
| **F1-score** | Precision 和 Recall 的调和平均 | 综合评估 |
| **Macro F1** | 各类别 F1 的算术平均 | 平等对待稀有类别 |

**关键目标：**
- Label 1 和 2 的 Recall > 0.6（能捕捉 60% 以上的放量机会）
- Label 1 和 2 的 Precision > 0.4（降低误判）

---

## 6. 完整防未来函数检查清单 ✅

在部署前，务必通过以下检查：

### ✅ 数据预处理阶段
- [ ] 数据已按 `time` 字段严格排序
- [ ] 训练/验证/测试集按时间顺序划分，无交叉
- [ ] 归一化仅使用训练集统计量，或使用滚动窗口
- [ ] 特征工程中所有指标均为因果计算（无 `shift(-n)`）

### ✅ 序列构建阶段
- [ ] 滑动窗口仅包含 `[T-seq_len+1, ..., T]` 的数据
- [ ] 标签对应时刻 T 的信息（非未来时刻）

### ✅ 训练阶段
- [ ] 验证集和测试集在时间上晚于训练集
- [ ] 早停监控的是验证集指标（非测试集）
- [ ] 过采样仅在训练集内进行

### ✅ 评估阶段
- [ ] 测试集仅在最终评估时使用一次
- [ ] 未使用测试集信息进行任何超参数调优

---

## 7. 实盘部署建议

### 7.1 模型推理流程

```python
def predict_current_signal(historical_data, model, scaler, seq_len=96):
    """
    为当前时刻生成交易信号
  
    Args:
        historical_data: 包含最近 seq_len 条 K 线的 DataFrame
        model: 训练好的 TimesNet 模型
        scaler: 训练时拟合的归一化器
        seq_len: 序列长度
  
    Returns:
        signal: 0 (无信号), 1 (放量上涨), 2 (放量下跌)
    """
    # 1. 特征工程（与训练时完全一致）
    historical_data = compute_technical_indicators(historical_data)
  
    # 2. 归一化（使用训练时的 scaler）
    features = historical_data[feature_list].values[-seq_len:]
    features_scaled = scaler.transform(features)
  
    # 3. 模型推理
    X = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)  # (1, seq_len, n_features)
  
    with torch.no_grad():
        output = model(X)
        signal = torch.argmax(output, dim=1).item()
  
    return signal
```

### 7.2 风险控制

- **置信度过滤**：仅在预测概率 > 0.6 时执行交易
- **止损机制**：即使是 Label 1/2 信号，也要设置止损点
- **信号确认**：结合其他指标（如成交量突增）进行二次确认

---

## 8. 附录：完整代码框架

```python
# main.py - 完整训练脚本

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ============ 1. 数据加载与预处理 ============
df = pd.read_csv('neiro_2h.csv')
df = df.sort_values('time').reset_index(drop=True)

# 时序划分
n = len(df)
train_df = df[:int(n*0.7)].copy()
val_df = df[int(n*0.7):int(n*0.85)].copy()
test_df = df[int(n*0.85):].copy()

# 因果归一化
feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
scaler = StandardScaler()
scaler.fit(train_df[feature_cols])

train_df[feature_cols] = scaler.transform(train_df[feature_cols])
val_df[feature_cols] = scaler.transform(val_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

# ============ 2. 特征工程 ============
def add_technical_indicators(df):
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
  
    # OBV
    df['obv'] = (df['volume'] * (~df['close'].diff().le(0) * 2 - 1)).cumsum()
  
    # ATR
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
  
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
  
    # ROC
    df['roc'] = df['close'].pct_change(12) * 100
  
    df.fillna(0, inplace=True)
    return df

train_df = add_technical_indicators(train_df)
val_df = add_technical_indicators(val_df)
test_df = add_technical_indicators(test_df)

# ============ 3. 序列构建 ============
feature_list = ['open', 'high', 'low', 'close', 'volume', 'amount',
                'rsi', 'obv', 'atr', 'macd', 'macd_signal', 'macd_hist', 'roc']

def create_sequences(df, seq_len=96):
    X, y = [], []
    data = df[feature_list].values
    labels = df['label'].values
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(labels[i])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_df)
X_val, y_val = create_sequences(val_df)
X_test, y_test = create_sequences(test_df)

# ============ 4. 数据加载器 ============
train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
    batch_size=64, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
    batch_size=64, shuffle=False
)

# ============ 5. 模型与损失函数 ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
  
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

alpha = torch.FloatTensor([1.0, 5.0, 5.0]).to(device)
criterion = FocalLoss(alpha=alpha, gamma=2.5)

# 模型（假设已实现）
model = TimesNet(seq_len=96, enc_in=13, c_out=3, task_name='classification').to(device)

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# ============ 6. 训练循环 ============
best_f1 = 0
patience_counter = 0

for epoch in range(100):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
      
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
      
        train_loss += loss.item()
  
    # 验证
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(y_batch.numpy())
  
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val F1={val_f1:.4f}")
  
    scheduler.step(val_f1)
  
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= 15:
            print("早停触发")
            break

# ============ 7. 测试评估 ============
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

X_test_t = torch.FloatTensor(X_test).to(device)
with torch.no_grad():
    test_preds = torch.argmax(model(X_test_t), dim=1).cpu().numpy()

print("\n测试集结果:")
print(classification_report(y_test, test_preds, target_names=['无信号', '放量上涨', '放量下跌']))
print("\n混淆矩阵:")
print(confusion_matrix(y_test, test_preds))
```

---

## 9. 总结与风险提示

### ✅ 本方案的核心优势
1. **零未来函数**：所有特征和归一化严格遵循因果性
2. **针对性设计**：Focal Loss + 类别权重专门解决不平衡问题
3. **可解释性**：技术指标具有金融意义（RSI、OBV、MACD）
4. **实盘友好**：评估指标聚焦 Recall，符合交易需求

### ⚠️ 风险声明
1. **过拟合风险**：虚拟货币市场非平稳性强，历史模式可能失效
2. **滑点与手续费**：实盘交易成本会显著降低收益
3. **黑天鹅事件**：模型无法预测突发新闻或政策冲击
4. **数据质量**：确保 K 线数据来源可靠且无缺失

### 🔧 后续优化方向
- **集成学习**：结合 Transformer、LSTM 等模型投票
- **注意力机制**：在 TimesNet 基础上加入 Temporal Attention
- **强化学习**：将信号转化为连续动作（持仓比例）
- **多资产联动**：加入 BTC、ETH 等相关币种的协同特征

---

**最终检查：在部署任何模型前，请使用真实历史数据进行"Walk-Forward Testing"（滚动回测），确保未来函数检查通过！**