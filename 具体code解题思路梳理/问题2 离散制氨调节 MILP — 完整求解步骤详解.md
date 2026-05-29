## 🤖 Assistant

# 问题二 离散制氨调节 MILP — 完整求解步骤详解

---

## 总体逻辑概览

Q2 的核心任务是：在设备只能**全开（0）或全关（1）**的离散约束下，针对 5 档日产量 $\times$ 24 种风光场景（共 120 个实例），构建并求解混合整数线性规划（MILP），找到每个实例成本最低的生产时段安排。

求解流程可以归纳为以下六个步骤：

```
步骤1: 容量扩容 → 将36t/d基准参数线性缩放至72t/d
步骤2: 变量定义 → 24时段 × 2类二元 + 2类连续 = 96个变量
步骤3: 约束构建 → 运行时长 + 功率平衡 + 购售电互斥(Big-M)
步骤4: 目标函数 → 购电费 + 运维 - 售电收入 + 绿电惩罚(软约束)
步骤5: CBC求解   → 120次MILP, 每次0.1-0.5秒
步骤6: 后处理   → 校验 + 分类统计 + 年度汇总 + 可视化
```

下面按数学逻辑逐层展开。

---

## 步骤1：从36吨到72吨的线性扩容

### 1.1 缩放系数

$$\alpha = \frac{72}{36} = 2.0$$

代码中：

```python
SCALE_72 = 72.0 / NH3_BASELINE_CAPACITY  # = 2.0
```

### 1.2 扩容后的设备参数

| 参数 | 36t/d 基准 | 72t/d 扩容 ($\times 2$) |
|------|:---:|:---:|
| ALKEL 功率 | 10 MW | **20 MW** |
| PEMEL 功率 | 10 MW | **20 MW** |
| NH₃ 功率 | 0.75 MW | **1.5 MW** |
| 设备总功率 $P_{\text{total}}$ | 20.75 MW | **41.5 MW** |
| ALKEL 制氢速率 | 140 kg/h | **280 kg/h** |
| PEMEL 制氢速率 | 160 kg/h | **320 kg/h** |
| 总制氢速率 | 300 kg/h | **600 kg/h** |
| NH₃ 产率 | 1.5 ton/h | **3.0 ton/h** |

### 1.3 H₂ 质量平衡验证（硬断言）

代码中有一个重要的自检：

$$600\ \text{kgH}_2/\text{h} \stackrel{?}{=} 3.0\ \text{ton/h} \times 1000\ \text{kg/ton} \times 0.2\ \text{kgH}_2/\text{kgNH}_3 = 600\ \text{kgH}_2/\text{h} \quad \checkmark$$

这确保了扩容后的制氢速率与合成氨耗氢速率精确匹配。

---

## 步骤2：MILP 决策变量定义

### 2.1 变量全景（每时段 $t = 0, 1, \ldots, 23$）

| 变量 | 类型 | 含义 | 取值范围 |
|------|:---:|------|:---:|
| $x_t$ | Binary | 设备运行标志 (1=ON, 0=OFF) | $\{0, 1\}$ |
| $g_t$ | Binary | 电网方向 (1=售电, 0=购电) | $\{0, 1\}$ |
| $\text{buy}_t$ | Continuous | 网购电功率 | $[0, +\infty)$ MW |
| $\text{sell}_t$ | Continuous | 上网电功率 | $[0, +\infty)$ MW |

> **关键设计**：Q2 采用**聚合设备建模**——ALKEL、PEMEL、NH₃ 被合并为单一总功率 $P_{\text{total}} = 41.5$ MW，由一个二元变量 $x_t$ 统一控制。这是与 Q3 拆分建模的根本区别。

**总变量数**：24 时段 $\times$ (2 binary + 2 continuous) = **48 binary + 48 continuous = 96 个变量**。

---

## 步骤3：约束条件构建

### 3.1 约束① — 运行时长约束（核心）

$$\boxed{\sum_{t=0}^{23} x_t = \left\lfloor \frac{\text{日产量}}{\text{满负荷产率}} \right\rfloor = \left\lfloor \frac{\text{日产量}}{3.0\ \text{ton/h}} \right\rfloor}$$

由于设备满负荷运行时产氨速率为 3.0 ton/h，完成日产量目标所需满负荷运行的小时数为：

| 日产量 (ton) | 运行小时 | 计算 |
|:---:|:---:|:---:|
| 72 | 24 h | $72/3.0 = 24$ |
| 63 | 21 h | $63/3.0 = 21$ |
| 54 | 18 h | $54/3.0 = 18$ |
| 45 | 15 h | $45/3.0 = 15$ |
| 36 | 12 h | $36/3.0 = 12$ |

代码实现：

```python
runtime_hours = int(daily_output / NH3_RATE_72)
prob += lpSum(x) == runtime_hours, "RuntimeHours"
```

> **注意**：代码中 $72/3.0 = 24$ 是精确整数，$63/3.0 = 21$ 也是精确整数。对于 72 吨/天，设备必然 24 小时全开 → **无任何调度自由度**。

### 3.2 约束② — 逐时功率平衡

$$\boxed{P_{\text{wind}}(t) + P_{\text{solar}}(t) + \text{buy}_t = \text{sell}_t + x_t \cdot P_{\text{total}} + P_{\text{load}}(t) \quad \forall t \in [0, 23]}$$

物理含义：

- **$x_t = 1$**：设备以 41.5 MW 满功率运行，缺口由电网补足、盈余上网
- **$x_t = 0$**：设备完全停机，风光出力要么供给常规负荷、要么上网

代码实现：

```python
for t in range(24):
    prob += (
        wind[t] + solar[t] + buy[t]
        == sell[t] + x[t] * P_TOTAL + load[t]
    ), f"Balance_{t}"
```

### 3.3 约束③ — 购售电互斥（Big-M 法）

这是 Q2 中最容易被忽视但**至关重要**的约束。

**问题**：如果允许同一时刻既购电又售电，MILP 会利用**谷时购电价（0.3424 元/kWh）< 上网电价（0.3779 元/kWh）**的价差进行无风险套利：谷时从电网买电、同时以更高价格卖回电网，数学上产生无界利润。

**解决方案**：引入方向指示变量 $g_t$，施加 Big-M 互斥约束：

$$\boxed{\begin{aligned} \text{buy}_t &\leq M \cdot (1 - g_t) \\ \text{sell}_t &\leq M \cdot g_t \end{aligned}}$$

其中 $M = P_{\text{wind}}^{\text{rated}} + P_{\text{solar}}^{\text{rated}} = 40 + 64 = 104$ MW。

- 当 $g_t = 0$：$\text{buy}_t \leq 104$（可购电），$\text{sell}_t \leq 0$（强制售电为 0）
- 当 $g_t = 1$：$\text{buy}_t \leq 0$（强制购电为 0），$\text{sell}_t \leq 104$（可售电）

代码实现：

```python
M_grid = config.WIND_RATED + config.SOLAR_RATED  # 104 MW
grid_dir = [LpVariable(f"grid_dir_{t}", cat="Binary") for t in range(24)]

for t in range(24):
    prob += buy[t] <= M_grid * (1 - grid_dir[t]), f"NoBuyWhenSell_{t}"
    prob += sell[t] <= M_grid * grid_dir[t], f"NoSellWhenBuy_{t}"
```

> **Big-M 的合理性论证**：104 MW 足以覆盖任何时刻的最大可能净功率（因为 $|P_{\text{net}}| \leq P_{\text{total}} + P_{\text{load}} \approx 47.5$ MW），约束不会过度限制可行域。

---

## 步骤4：目标函数构建

### 4.1 目标函数结构

$$\min \quad \underbrace{\sum_{t=0}^{23} \text{buy}_t \cdot c_t^{\text{TOU}} \cdot 1000}_{\text{购电费（分时电价）}} + \underbrace{T_{\text{run}} \cdot P_{\text{total}} \cdot 1000 \cdot \bar{c}_{\text{om}}}_{\text{电解运维费（固定）}} + \underbrace{T_{\text{run}} \cdot P_{\text{NH}_3} \cdot 1000 \cdot c_{\text{NH}_3}^{\text{om}}}_{\text{合成氨运维费（固定）}} - \underbrace{\sum_{t=0}^{23} \text{sell}_t \cdot 0.3779 \cdot 1000}_{\text{售电收入}} + \underbrace{P_{\text{green}}}_{\text{绿电惩罚}}$$

### 4.2 各项详解

**购电费**：使用 config.py 中的分时电价查找表

$$c_t^{\text{TOU}} = \begin{cases} 0.3424 & t \in [23, 0-6] \text{ (低谷)} \\ 0.6074 & t \in [7-9, 15-17, 21-22] \text{ (平段)} \\ 0.8024 & t \in [10-14, 18-20] \text{ (高峰)} \end{cases}$$

代码中 `buy[t] * config.get_price(t) * 1000`：`buy[t]` 单位为 MW，×1000 转换为 kW，再乘以电价（元/kWh）。

**运维费**：Q2 使用混合平均运维费率

$$\bar{c}_{\text{om}} = \frac{0.10 + 0.15}{2} = 0.125\ \text{元/kWh}$$

电解运维总费用 = $T_{\text{run}} \times 41.5 \times 1000 \times 0.125$（由运行小时数确定，非决策变量）。

> **这是 Q2 与 Q3 的一个重要差异**：Q2 运维费按"运行小时 × 总功率 × 平均费率"计算，与调度方案无关（只要运行小时数固定）；Q3 拆分设备后，运维费取决于各设备实际用电量，成为优化目标的一部分。

**售电收入**：$\text{sell}_t \times 0.3779 \times 1000$（上网电价统一为 0.3779 元/kWh）。

### 4.3 绿电惩罚（软约束）

绿电三指标不直接作为硬约束（否则可能无可行解），而是通过**惩罚函数**纳入目标：

$$P_{\text{green}} = \underbrace{0.5}_{\lambda_1} s_{\text{self}} + \underbrace{0.3}_{\lambda_2} s_{\text{green}} + \underbrace{1.5}_{\lambda_3} s_{\text{feed}}$$

其中三个松弛变量 $s_{\text{self}}, s_{\text{green}}, s_{\text{feed}} \geq 0$ 度量偏离政策阈值的程度：

$$\begin{aligned} s_{\text{self}} &\geq 0.60 \cdot E_{\text{RE}} - (E_{\text{load}} - E_{\text{sell}} - E_{\text{buy}}) \\ s_{\text{green}} &\geq 0.30 \cdot E_{\text{load}} - (E_{\text{RE}} - E_{\text{sell}}) \\ s_{\text{feed}} &\geq E_{\text{sell}} - 0.20 \cdot E_{\text{RE}} \end{aligned}$$

> **$\lambda_3 = 1.5$ 最高**的设定逻辑：上网比例超标意味着绿电外送套利，需要最强的惩罚力度来抑制这种行为。$\lambda_1 = 0.5$ 对应网购电价中值，$\lambda_2 = 0.3$ 对应绿证+碳价。

---

## 步骤5：CBC 求解器求解

```python
prob.solve(PULP_CBC_CMD(msg=False))
```

每个 MILP 实例的特点：
- **变量规模**：48 个二元变量 + 48 个连续变量 + 绿电松弛变量
- **约束规模**：约 100 条线性约束（24 条功率平衡 + 48 条 Big-M + 1 条运行时长 + 绿电惩罚约束）
- **求解时间**：0.1–0.5 秒（CBC 开源求解器，分支定界法）

---

## 步骤6：后处理与校验

### 6.1 三层校验

求解完成后，`_verify_solution()` 对结果进行三层校验：

**第一层 — 功率平衡校验**（容差 $10^{-4}$ MW）：

$$|P_{\text{wind}}(t) + P_{\text{solar}}(t) + \text{buy}_t - \text{sell}_t - x_t \cdot 41.5 - P_{\text{load}}(t)| < 10^{-4} \quad \forall t$$

**第二层 — H₂ 质量平衡**：

$$600 \cdot T_{\text{run}} \stackrel{?}{=} \text{日产量} \times 1000 \times 0.2$$

**第三层 — 绿电指标范围检查**（$\eta \in [0, 1]$），极端场景下可能超出，仅发 WARNING。

### 6.2 吨氨成本后评估

注意：目标函数最小化的是"日运行成本"（不含风光发电成本和设备折旧），而吨氨成本的计算在 `calc_ton_ammonia_cost()` 中完成，额外加入：

- 风光度电成本：$E_{\text{wind}} \times 0.15 + E_{\text{solar}} \times 0.12$
- 设备折旧：通过 `calc_total_depreciation(daily_output)` 按产量分摊

$$C_{\text{ton}} = \frac{\text{购电费} + \text{风光成本} + \text{运维} + \text{折旧} - \text{售电收入}}{\text{日产量}}$$

### 6.3 年度统计

采用 **24 场景 × 15 天/场景 = 360 天/年** 的加权方案：

$$\text{加权平均吨氨成本} = \frac{\sum_{\text{all } 120 \text{ results}} C_{\text{ton}}^{(i)} \times 15}{\sum_{\text{all } 120 \text{ results}} 15} = \frac{\sum C_{\text{ton}}^{(i)}}{120}$$

分类统计：

$$\begin{aligned} N_{\text{全满足}} &= \#\{\eta_1 > 0.60 \land \eta_2 > 0.30 \land \eta_3 < 0.20\} \times 15 \\ N_{\text{部分满足}} &= \#\{\text{含但不全}\} \times 15 \\ N_{\text{全不满足}} &= \#\{\text{三项均不合格}\} \times 15 \end{aligned}$$

---

## 第一部分（典型场景）的求解流程

对于典型场景（附件 2），遍历 5 个产量级别：

```
for daily_output in [72, 63, 54, 45, 36]:
    → build_and_solve_milp(wind, solar, load, daily_output)
    → 记录: 运行时段 x[t], 购售电 buy[t]/sell[t], 吨氨成本
    → 比较 5 个结果，取吨氨成本最低的日产量
```

关键发现：
- 72 吨/天时 `runtime_hours = 24` → 设备必须 24h 全开 → **零调度自由度**
- 36 吨/天时 `runtime_hours = 12` → MILP 可在 24 个时段中**选择最优的 12 个小时**开机

### 最优产量分析

MILP 自动选择策略：**在低电价时段（谷时 0.3424）集中运行，避免高电价时段（峰时 0.8024）购电**。求解器通过比较各时段购电成本（$\text{buy}_t \times c_t^{\text{TOU}}$）来自动决策。

---

## 第二部分（全部 24 场景）的求解流程

```
for (w, s) in 24 scenarios:          # 风电 0-5 × 光伏 0-3
    wind  = get_wind_profile(w)       # (24,) MW
    solar = get_solar_profile(s)      # (24,) MW
    for daily_output in [72,63,54,45,36]:
        → build_and_solve_milp(...)
        → 计算绿电三指标
        → classify_scenario(indicators)  → "全满足"/"部分满足"/"全不满足"
        → 记录到 all_results[]
```

### 场景分类逻辑

```python
def classify_scenario(indicators):
    self_ok = indicators['self_use_rate'] > 0.60
    green_ok = indicators['green_rate'] > 0.30
    feed_ok = indicators['grid_feed_rate'] < 0.20
    
    if self_ok and green_ok and feed_ok:  return "全满足"
    if self_ok or green_ok or feed_ok:    return "部分满足"
    return "全不满足"
```

---

## 求解结果汇总

| 维度 | 数值 |
|------|:---:|
| 求解成功率 | 120/120 (100%) |
| 典型场景最优产量 | **72 吨/天**（满产，规模效应压倒一切） |
| 典型场景 W0S0 72t/d 吨氨成本 | ~5447 元/吨 |
| 年度加权平均吨氨成本 | ~5943 元/吨 |
| 全满足天数 | ~120 天（33%） |
| 部分满足 + 全不满足 | ~240 天（67%） |

### 离散调节的核心局限

1. **满产时无自由度**（72t/d → 24h 全开 = 退回 Q1 固定运行）
2. **中低产量时**虽然可以选择运行时段，但由于设备只能全功率（41.5 MW），无法降低功率来避开高电价 → 购电成本压缩空间有限
3. **运维费与调度脱钩**（按运行小时计算，而非实际用电量）

这正是 Q3（连续调节 + 拆分建模）需要改进的关键方向。
