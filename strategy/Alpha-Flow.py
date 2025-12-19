import pandas as pd
import numpy as np

class AlphaFlowStrategy:
    def __init__(self, initial_capital=10000, risk_pct=0.02):
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct  # 单笔交易承担总资金的 2% 风险
        self.params = {
            'n': 20,            # 周期
            't_bull': 0.40,     # 做多阈值 (严选)
            't_bear': -0.45,    # 做空阈值 (严选)
            'atr_mult': 3.0     # 止损宽度 (3倍ATR)
        }

    def calculate_indicators(self, df):
        """计算 vRDI 因子和 ATR"""
        df = df.copy()
        n = self.params['n']
        
        # 1. 处理成交量 (Tick Volume 或 Volume)
        vol_col = 'tick_volume' if 'tick_volume' in df.columns else 'volume'
        vol = df[vol_col].replace(0, np.nan).fillna(method='ffill')
        
        # 2. 计算 ATR
        prev_close = df['close'].shift(1)
        prev_close.iloc[0] = df['open'].iloc[0]
        h_l = df['high'] - df['low']
        h_pc = (df['high'] - prev_close).abs()
        l_pc = (df['low'] - prev_close).abs()
        tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        df['atr'] = tr.rolling(n).mean()
        
        # 3. 计算 vRDI 因子
        vol_ma = vol.rolling(n).mean()
        rv = vol / (vol_ma + 1e-8)
        # 分母加入 ATR 防止微小波动的噪音
        denominator = h_l + df['atr']
        # 核心公式
        df['vrdi'] = ((df['close'] - df['open']) / denominator) * np.sqrt(rv)
        
        return df

    def run_backtest(self, df):
        """执行回测"""
        data = self.calculate_indicators(df)
        capital = self.initial_capital
        equity_curve = [capital] * len(data)
        
        position = 0        # 1: Long, -1: Short
        entry_price = 0.0
        stop_loss = 0.0
        size = 0.0          # 持仓数量
        
        # 提取 Numpy 数组加速循环
        opens = data['open'].values
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        atrs = data['atr'].values
        vrdis = data['vrdi'].values
        dates = data.index
        
        signals = np.zeros(len(data)) # 记录信号
        
        # 从第 N 个数据开始遍历
        start_idx = self.params['n']
        
        for i in range(start_idx, len(data)):
            # 获取当前 K 线 (t) 的数据
            curr_open = opens[i]
            curr_vrdi = vrdis[i-1] # 信号基于上一根K线收盘 (t-1)
            curr_atr = atrs[i-1]
            
            # --- 1. 资金结算 (Mark to Market) ---
            # 简单的逐日盯市权益计算
            curr_equity = capital
            if position == 1:
                curr_equity += size * (closes[i] - entry_price)
            elif position == -1:
                curr_equity += size * (entry_price - closes[i])
            
            # --- 2. 检查止损 (Intra-bar Stop Loss) ---
            trade_closed = False
            if position == 1:
                if lows[i] < stop_loss: # 触及止损
                    exit_price = min(opens[i], stop_loss) # 如果开盘就低开，按开盘价止损
                    pnl = size * (exit_price - entry_price)
                    capital += pnl
                    position = 0
                    trade_closed = True
            elif position == -1:
                if highs[i] > stop_loss:
                    exit_price = max(opens[i], stop_loss)
                    pnl = size * (entry_price - exit_price)
                    capital += pnl
                    position = 0
                    trade_closed = True

            # --- 3. 移动止损更新 (Trailing) ---
            if position == 1 and not trade_closed:
                new_sl = closes[i] - (self.params['atr_mult'] * atrs[i])
                if new_sl > stop_loss: stop_loss = new_sl
            elif position == -1 and not trade_closed:
                new_sl = closes[i] + (self.params['atr_mult'] * atrs[i])
                if new_sl < stop_loss: stop_loss = new_sl

            # --- 4. 生成新信号 (基于上一根K线的 vRDI) ---
            # 如果我们空仓，或者是反转信号
            target_pos = position
            
            if curr_vrdi > self.params['t_bull']:
                target_pos = 1
            elif curr_vrdi < self.params['t_bear']:
                target_pos = -1
            
            # --- 5. 执行交易 (在当前 Open 执行) ---
            # 实际上我们是在 t 时刻看到了 t-1 的 vRDI，决定在 t 时刻 Open 介入
            # 为了简化，我们假设如果上一根触发信号，我们已经在 Open 进场了
            # 这里做逻辑修正：信号触发是在 t-1 收盘，交易是在 t 开盘
            
            if not trade_closed and target_pos != position:
                # 平掉旧仓
                if position != 0:
                    if position == 1: pnl = size * (opens[i] - entry_price)
                    else: pnl = size * (entry_price - opens[i])
                    capital += pnl
                
                # 开新仓 (Position Sizing)
                risk_amt = capital * self.risk_pct
                sl_dist = self.params['atr_mult'] * curr_atr
                if sl_dist == 0: sl_dist = opens[i] * 0.01 # 防止除零
                
                new_size = risk_amt / sl_dist
                
                position = target_pos
                size = new_size
                entry_price = opens[i]
                
                # 初始止损
                if position == 1: stop_loss = entry_price - sl_dist
                else: stop_loss = entry_price + sl_dist
            
            equity_curve[i] = curr_equity
            signals[i] = position

        data['equity'] = equity_curve
        data['strategy_pos'] = signals
        return data

# 使用示例
# strategy = AlphaFlowStrategy()
# res = strategy.run_backtest(df)
# print(f"Final Equity: {res['equity'].iloc[-1]:.2f}")
