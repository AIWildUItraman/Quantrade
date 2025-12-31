# BNBUSDT K线数据标注工具 - 执行计划

## 任务概述

**目标**：创建一个基于HTML的单页应用，用于标注BNBUSDT K线数据

**技术方案**：纯前端 Canvas + Vanilla JavaScript（无外部依赖）

**交付物**：`crypto_timesnet/annotation_tool.html` 单文件应用

---

## 核心需求

### 输入
- CSV文件（包含timestamp, open, high, low, close, volume字段）

### 数据处理
- 滑动窗口：每24根K线为一组
- 归一化：对每组数据进行 [0,1] 归一化
- 可视化：Canvas绘制归一化K线图

### UI结构
- 三图展示区：左（上一张）+ 中（当前标注）+ 右（下一张）
- 分类按钮：上涨 / 下跌 / 波动
- 导航按钮：上一张 / 下一张
- 进度显示：标注进度条 + 分类统计
- 实时保存：每次标注后立即保存CSV

### 交互逻辑
- 点击分类按钮 → 保存标注 → 自动跳转下一张
- 点击导航按钮 → 查看/修改历史标注
- 键盘快捷键：左右箭头翻页，数字键1/2/3快速标注

---

## 架构设计

### 核心模块

1. **DataProcessor（数据处理模块）**
   - CSV解析
   - 滑动窗口切分
   - 数据归一化算法

2. **ChartRenderer（Canvas渲染模块）**
   - K线图绘制
   - 网格和坐标轴
   - 标注状态高亮

3. **AnnotationManager（标注管理模块）**
   - 标注状态维护
   - 统计信息计算
   - CSV导出

4. **UIController（UI控制模块）**
   - 页面跳转
   - 三图联动
   - 事件绑定

---

## 详细执行步骤

### Step 1-2: HTML结构与CSS样式
- 创建基础HTML5结构
- 三图Flexbox布局（左右400x300，中间600x400）
- 按钮样式：绿（上涨）、红（下跌）、橙（波动）
- 进度条渐变背景

### Step 3-5: 数据处理流程
- 文件上传与FileReader读取
- CSV解析（提取关键字段）
- 滑动窗口切分（24根K线）
- 归一化算法：`(value - min) / (max - min)`

### Step 6-8: Canvas渲染系统
- 初始化三个Canvas元素
- K线绘制核心算法
  - 阳线：绿色实心（close > open）
  - 阴线：红色空心（close < open）
  - 上下影线
- 三图联动渲染

### Step 9-12: 交互与保存
- 标注按钮点击 → 保存 → 自动跳转
- 导航按钮翻页
- 进度统计实时更新
- CSV自动下载保存

### Step 13-14: 增强与优化
- 键盘快捷键
- 样式细节调整
- 边界测试

---

## 技术要点

### 归一化公式
```javascript
normalized_value = (value - min) / (max - min)
canvas_y = (1 - normalized_value) * canvas_height
```

### K线坐标计算
```javascript
x = (index / 24) * canvas.width
candleWidth = canvas.width / 24 * 0.8
bodyTop = (1 - norm_close) * height
bodyBottom = (1 - norm_open) * height
```

### CSV导出格式
```csv
window_index,label,time_0,open_0,high_0,low_0,close_0,volume_0,...,time_23,...
0,up,timestamp1,price1,...
1,down,timestamp2,...
```

---

## 风险与注意事项

⚠️ **浏览器兼容性**：使用现代浏览器（Chrome/Firefox/Edge）
⚠️ **文件大小**：建议CSV文件 < 50MB
⚠️ **CSV格式**：必须包含完整的OHLCV字段

---

## 执行状态

- [x] 计划制定完成
- [ ] HTML结构搭建
- [ ] CSS样式编写
- [ ] 数据处理模块
- [ ] Canvas渲染模块
- [ ] 交互功能实现
- [ ] 测试与优化

**执行时间**：2025-12-30
**状态**：执行中
