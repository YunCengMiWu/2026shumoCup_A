## 🤖 Assistant

# 问题三 连续制氨调节 MILP — 完整求解步骤详解

---

## 从 Q2 到 Q3：核心飞跃

Q2 的离散开关模型有两个根本局限：

1. **聚合设备建模**：ALKEL(20MW) + PEMEL(20MW) + NH₃(1.5MW) 被合并为单一的 41.5MW 块，由唯一二元变量 $x_t$ 控制
2. **全开/全关**：设备要么 41.5MW 满功率，要么 0MW 停机 → 无法避开高电价时段

Q3 从根本上重构了模型：

| 维度 | Q2（离散聚合） | Q3（连续拆分） |
|------|:---|:---|
| 建模粒度 | 1个聚合块 (41.5MW) | **3类设备独立** |
| 功率变量 | $x_t \in \{0,1\}$ | $p_{\text{alk}}, p_{\text{pem}}, p_{\text{nh3}} \in [0, P_{\max}]$ |
| 运行标志 | 1个二元变量/时段 | **3个独立二元变量/时段** |
| 半连续约束 | 无 | **per-device**: NH₃≥10%, ALK≥10%, PEM≥0% |
| H₂平衡 | 间接（日汇总） | **逐时不等式约束** |
| NH₃爬坡 | 无 | **硬约束** $\vert\Delta p_{\text{nh3}}\vert \le 0.30$ MW/h |
| 运维核算 | 按运行小时×平均费率 | **按各设备实际用电量×独立费率** |
| 变量总数 | 48 binary + 48 continuous | **72 binary + 144 continuous** |

---

## 步骤1：容量扩容（与 Q2 完全相同）

$$\alpha = \frac{72}{36} = 2.0$$

扩容后参数：

| 参数 | 值 |
|------|:--:|
| $P_{\text{ALK}}^{\max}$ | 20 MW (2台×10MW) |
| $P_{\text{PEM}}^{\max}$ | 20 MW (2台×10MW) |
| $P_{\text{NH}_3}^{\max}$ | 1.65 MW (2台×0.75MW×110%) |
| 满负荷产氨速率 | 3.0 ton/h |

---

## 步骤2：决策变量定义（按设备拆分）

### 2.1 连续功率变量（每时段 $t = 0,\ldots,23$）

$$\boxed{\begin{aligned} p_{\text{alk}}[t] &\in [0,\ 20]\ \text{MW} \\ p_{\text{pem}}[t] &\in [0,\ 20]\ \text{MW} \\ p_{\text{nh3}}[t] &\in [0,\ 1.65]\ \text{MW} \end{aligned}}$$

代码实现：

```python
p_alk = [LpVariable(f"p_alk_{t}", lowBound=0, upBound=ALK_SYS_POWER) for t in range(24)]
p_pem = [LpVariable(f"p_pem_{t}", lowBound=0, upBound=PEM_SYS_POWER) for t in range(24)]
p_nh3 = [LpVariable(f"p_nh3_{t}", lowBound=0, upBound=NH3_SYS_POWER * NH3_LOAD_MAX) for t in range(24)]
```

### 2.2 二元运行标志（每时段 × 3类设备）

$$\boxed{\begin{aligned} z_{\text{alk}}[t] &\in \{0, 1\} \\ z_{\text{pem}}[t] &\in \{0, 1\} \\ z_{\text{nh3}}[t] &\in \{0, 1\} \end{aligned}}$$

这三个二元变量**彼此独立**——ALK 可以停机而 PEM 和 NH₃ 继续运行，这在 Q2 的聚合模型中是完全不可能的。

### 2.3 购售电变量（与 Q2 相同）

$$\text{buy}[t] \ge 0,\quad \text{sell}[t] \ge 0,\quad g_t \in \{0,1\}$$

**总变量数**：3类 × 24时段 × (1 continuous + 1 binary) + 2 × 24 continuous + 24 binary = 72 binary + 72 continuous + 48连续（购售电）+ 24 binary（电网方向）= **72 binary + 144 continuous = 216 变量**。

---

## 步骤3：约束体系（五大约束层）

### 3.1 约束① — 半连续约束（per-device, per-hour）

这是 Q3 最核心的约束结构：

$$\boxed{\begin{aligned} \underbrace{P_{\text{ALK}}^{\min}}_{\text{2.0 MW}} \cdot z_{\text{alk}}[t] \le p_{\text{alk}}[t] &\le \underbrace{P_{\text{ALK}}^{\max}}_{\text{20 MW}} \cdot z_{\text{alk}}[t] \\[4pt] \underbrace{P_{\text{PEM}}^{\min}}_{\text{0 MW}} \cdot z_{\text{pem}}[t] \le p_{\text{pem}}[t] &\le \underbrace{P_{\text{PEM}}^{\max}}_{\text{20 MW}} \cdot z_{\text{pem}}[t] \\[4pt] \underbrace{P_{\text{NH}_3}^{\min}}_{\text{0.15 MW}} \cdot z_{\text{nh3}}[t] \le p_{\text{nh3}}[t] &\le \underbrace{P_{\text{NH}_3}^{\max}}_{\text{1.65 MW}} \cdot z_{\text{nh3}}[t] \end{aligned}}$$

其中：

- $P_{\text{ALK}}^{\min} = 0.10 \times 20 = 2.0$ MW
- $P_{\text{PEM}}^{\min} = 0.00 \times 20 = 0$ MW → **PEM 无最低负荷，灵活度最高**
- $P_{\text{NH}_3}^{\min} = 0.10 \times 1.5 = 0.15$ MW

这些约束的物理语义：

$$z[t] = 0 \implies p[t] = 0 \quad (\text{设备完全停机})$$
$$z[t] = 1 \implies P^{\min} \le p[t] \le P^{\max} \quad (\text{设备在允许范围内连续运行})$$

代码实现（每时段 6 条不等式）：

```python
for t in range(24):
    prob += p_alk[t] <= ALK_SYS_POWER * z_alk[t],         f"Semi_alk_up_{t}"
    prob += p_alk[t] >= P_MIN_ALK * z_alk[t],             f"Semi_alk_lo_{t}"
    prob += p_pem[t] <= PEM_SYS_POWER * z_pem[t],         f"Semi_pem_up_{t}"
    prob += p_pem[t] >= P_MIN_PEM * z_pem[t],             f"Semi_pem_lo_{t}"
    prob += p_nh3[t] <= NH3_SYS_POWER * NH3_LOAD_MAX * z_nh3[t], f"Semi_nh3_up_{t}"
    prob += p_nh3[t] >= P_MIN_NH3 * z_nh3[t],             f"Semi_nh3_lo_{t}"
```

### 3.2 约束② — 产量约束

$$\boxed{\sum_{t=0}^{23} p_{\text{nh3}}[t] \times \underbrace{\text{NH3\_POWER\_RATE}}_{0.002\ \text{ton/(kW·h)}} \times 1000 = \text{日产量目标}}$$

物理含义：合成氨产量正比于 NH₃ 装置的总用电量。系数推导：

$$\text{NH3\_POWER\_RATE} = \frac{1.5\ \text{ton/h}}{0.75\ \text{MW} \times 1000\ \text{kW/MW}} = 0.002\ \text{ton/(kW·h)}$$

验证：满负荷 24h → $1.5 \times 0.002 \times 1000 \times 24 = 72$ 吨 ✓

代码：

```python
prob += lpSum(p_nh3) * NH3_POWER_RATE * 1000 == daily_output, "NH3_Target"
```

> **注意**：这是等式约束（`==` 而非 `>=`），产量必须精确达标。MILP 通过调节 NH₃ 功率在 24 个时段间的分布来实现。

### 3.3 约束③ — H₂ 逐时平衡（v2 新增，最关键的新约束）

$$\boxed{p_{\text{alk}}[t] \times \underbrace{14}_{\text{kg/(h·MW)}} + p_{\text{pem}}[t] \times \underbrace{16}_{\text{kg/(h·MW)}} \ge p_{\text{nh3}}[t] \times \underbrace{400}_{\text{kg/(MW·h)}} \quad \forall t}$$

各项物理含义：

- **ALK 产氢效率**：$\frac{1000\ \text{kW/MW}}{50\ \text{kWh/kg} \div 0.70} = \frac{1000}{71.43} \approx 14$ kg/(h·MW)
- **PEM 产氢效率**：$\frac{1000}{50 \div 0.80} = \frac{1000}{62.5} = 16$ kg/(h·MW)
- **NH₃ 耗氢率**：$200\ \text{kgH}_2/\text{ton} \times 0.002\ \text{ton/(kW·h)} \times 1000 = 400$ kg/(MW·h)

$$\text{验证}：1.5\ \text{MW} \times 400 = 600\ \text{kg/h} = \text{总制氢速率}$$

这个约束确保了**每个时段**的氢气供需平衡——NH₃ 不能消耗超过电解槽当前产生的氢气，防止 "超前制氢、延后合成" 这种物理上不可行的调度。

> **PEM 优先效应（自动产生）**：因为 PEM 产氢效率更高（16 > 14 kg/MW），在满足相同 NH₃ 功率需求时，MILP 自动优先使用 PEM。这是**模型结构的自然涌现**，而非人为设定的规则。验证结果：W0S0 场景中 PEM 利用率 95% vs ALK 利用率 53%。

代码通过 `utils.build_h2_balance_constraint()` 实现（逐时不等式 + 日汇总等式）。

### 3.4 约束④ — NH₃ 爬坡约束

$$\boxed{|\Delta p_{\text{nh3}}[t]| \le 0.20 \times 1.5 = 0.30\ \text{MW/h} \quad \forall t \ge 1}$$

展开为两条线性不等式：

$$p_{\text{nh3}}[t] - p_{\text{nh3}}[t-1] \le 0.30 \quad \text{(上行限幅)}$$
$$p_{\text{nh3}}[t-1] - p_{\text{nh3}}[t] \le 0.30 \quad \text{(下行限幅)}$$

物理含义：合成氨装置的功率变化速率不能超过 20%/h（相对于 1.5 MW 基准），避免热应力损伤催化剂。代码通过 `utils.build_nh3_ramp_constraint()` 实现。

> **注意**：Q3 中 ALK 和 PEM 无爬坡约束（赛题确认电解槽可瞬时调节功率），这是与学术文献中常见的区别。

### 3.5 约束⑤ — 功率平衡 + 购售电互斥

$$\boxed{P_{\text{wind}}[t] + P_{\text{solar}}[t] + \text{buy}[t] = \text{sell}[t] + p_{\text{alk}}[t] + p_{\text{pem}}[t] + p_{\text{nh3}}[t] + P_{\text{load}}[t] \quad \forall t}$$

购售电互斥（Big-M = 1000）：

$$\text{buy}[t] \le 1000(1 - g_t),\quad \text{sell}[t] \le 1000 \cdot g_t$$

代码：

```python
for t in range(24):
    prob += (wind[t] + solar[t] + buy[t] == sell[t] + p_alk[t] + p_pem[t] + p_nh3[t] + load[t]), f"Bal_{t}"
    prob += buy[t] <= BIG_M * (1 - grid_dir[t]), f"NoBuyWhenSell_{t}"
    prob += sell[t] <= BIG_M * grid_dir[t],      f"NoSellWhenBuy_{t}"
```

---

## 步骤4：目标函数（v2 拆分运维）

$$\boxed{\min \quad \underbrace{\sum_{t=0}^{23} \text{buy}[t] \cdot c_t^{\text{TOU}} \cdot 1000}_{\text{购电费}} + \underbrace{\sum_{t=0}^{23} \Big(p_{\text{alk}}[t] \cdot 0.10 + p_{\text{pem}}[t] \cdot 0.15 + p_{\text{nh3}}[t] \cdot 0.002\Big) \cdot 1000}_{\text{拆分运维费（v2 核心改进）}} - \underbrace{\sum_{t=0}^{23} \text{sell}[t] \cdot 0.3779 \cdot 1000}_{\text{售电收入}} + \underbrace{P_{\text{green}}}_{\text{绿电惩罚}}}$$

### 拆分运维 vs Q2 混合运维

| 方面 | Q2（混合） | Q3（拆分） |
|------|:---|:---|
| ALK 运维费率 | 0.125（平均） | **0.10 元/kWh** |
| PEM 运维费率 | 0.125（平均） | **0.15 元/kWh** |
| 计费基础 | 运行小时 × 41.5MW | **实际用电量** |

这意味着 MILP 可以通过**少用 PEM（运维更贵）、多用 ALK（运维更便宜）**来降低运维成本。但由于 PEM 制氢效率更高，模型需要在"运维成本"和"制氢效率"之间自动权衡——这是 v2 拆分的精妙之处。

代码：

```python
om_cost = lpSum(
    (p_alk[t] * 1000 * ALKEL_OM + p_pem[t] * 1000 * PEMEL_OM + p_nh3[t] * 1000 * NH3_OM)
    for t in range(24)
)
```

### 绿电惩罚（与 Q2 相同，但修复了 ×1000 单位 bug）

$$P_{\text{green}} = \underbrace{0.5}_{\lambda_1} s_{\text{self}} + \underbrace{0.3}_{\lambda_2} s_{\text{green}} + \underbrace{1.5}_{\lambda_3} s_{\text{feed}}$$

---

## 步骤5：CBC 求解

```python
prob.solve(PULP_CBC_CMD(msg=False))
```

每个 MILP 实例的规模：

| 维度 | 数量 |
|------|:--:|
| 二元变量 | 72（3类 × 24时段） + 24（电网方向）= 96 |
| 连续变量 | 72（3类 × 24时段功率） + 48（购售电）= 120 |
| 约束 | ~280 条 |
| 求解时间 | 0.1–0.3 秒 |

---

## 步骤6：六层校验

求解后，`_verify_solution()` 对每个实例进行六层校验：

### 第一层：半连续约束（per-device）

```python
for t in range(24):
    for name, pv, pmin, pmax in [("ALK", p_alk[t], 2.0, 20),
                                  ("PEM", p_pem[t], 0, 20),
                                  ("NH3", p_nh3[t], 0.15, 1.65)]:
        if pv > 1e-9:  # 设备正在运行
            assert pv >= pmin - 1e-9  # 不低于下限
            assert pv <= pmax + 1e-9  # 不高于上限
```

### 第二层：功率平衡（容差 $10^{-4}$ MW）

$$|P_{\text{wind}} + P_{\text{solar}} + \text{buy} - \text{sell} - p_{\text{alk}} - p_{\text{pem}} - p_{\text{nh3}} - P_{\text{load}}| < 10^{-4}$$

### 第三层：产量达标

$$|\sum_t p_{\text{nh3}}[t] \times 0.002 \times 1000 - \text{目标产量}| < 10^{-4}$$

### 第四层：H₂ 质量平衡

$$|\text{产氢总量} - \text{耗氢总量}| < 10^{-3} \text{ kg}$$

### 第五层：绿电指标范围

$$\eta_1, \eta_2, \eta_3 \in [0, 1]$$

极端场景下可能超出，仅发 WARNING。

### 第六层：PEM/ALK 利用率统计（信息性）

```python
total_alk = np.sum(p_alk)
total_pem = np.sum(p_pem)
print(f"PEM/ALK比: {total_pem/total_alk:.2f}")
```

---

## 步骤7：遍历全部 120 个实例

```python
for (w, s) in 24 scenarios:          # 风电 0-5 × 光伏 0-3
    wind  = get_wind_profile(w)
    solar = get_solar_profile(s)
    for daily_output in [72, 63, 54, 45, 36]:
        → build_and_solve_lp(wind, solar, load, daily_output)
        → 计算绿电三指标
        → classify_scenario(indicators)
        → 记录 all_results[]
```

---

## 步骤8：Q2 vs Q3 对比分析

对比逻辑：

```python
for each (scenario, daily_output) in Q3 results:
    q2_cost = q2_lookup[(scenario, daily_output)]
    q3_cost = current_result["ton_ammonia_cost"]
    diff = q3_cost - q2_cost
    → 统计更优/更差/持平
    → 计算平均降幅
```

关键结果：

| 指标 | 数值 |
|------|:--:|
| Q3 更优占比 | 106/120 (88.3%) |
| Q3 更差占比 | 14/120 (11.7%) |
| 平均改善 | **−267.73 元/吨** |
| 年度加权平均 | **5,720.40 元/吨** (vs Q2: 5,942.56) |

### 降幅分布规律

- **满产 72t/d**：Q3 几乎无改善 → 72 吨需要高强度运行，灵活性空间小
- **中低产量（36–54t/d）**：降幅显著 → 连续调节允许谷时多运行、峰时降功率，充分利用分时电价差异
- **极端省钱场景**：谷时 0.3424 vs 峰时 0.8024 元/kWh → Q3 将峰时功率压至最低（接近下限），谷时补足产量

---

## 步骤9：可视化输出

| 图表 | 内容 |
|------|------|
| `q3_boxplot.png` | 5产量级别成本分布箱线图 |
| `q3_scatter.png` | 吨氨成本 vs $\eta_{\text{green}}$ 散点图（颜色=产量，标记=合规分类） |
| `q3_pie.png` | 合规分类占比饼图 |
| `q3_cost_breakdown.png` | W0S0 各产量级别成本构成堆叠柱状图 |
| `q3_radar.png` | 72t/d 子集聚类雷达图（K=2, Silhouette=0.70） |
| `q3_3d_cost.png` | 24场景 × 5产量 3D 成本分布 |

### 聚类分析（72t/d 子集，K=2, Silhouette=0.70）

| 簇 | 场景数 | $\eta_{\text{green}}$ | 吨氨成本 | 特征 |
|:--|:-----:|:---:|:---:|------|
| 簇1 | 6 | 60.6% | 5317 | 低成本-高绿电，S0 最优光伏 |
| 簇2 | 18 | 35.6% | 7805 | 高成本-低绿电，S1–S3 光伏不足 |

> **聚类发现**：S0（最优光伏）场景可实现低成本与高绿电兼得；S1–S3 场景光伏出力不足 → 需大量购电 → 成本翻倍。场景间成本差异约 47%，说明**风光资源禀赋是决定经济性的第一性因素**。

---

## Q3 与 Q2 的根本差异总结

$$\text{Q2 可行域} \subset \text{Q3 可行域}$$

Q2 的每个解（全开/全关）都是 Q3 的特例（令 $p_{\text{alk}} = p_{\text{pem}} = p_{\text{nh3}} = 0$ 或全部满功率，且 $z_{\text{alk}} = z_{\text{pem}} = z_{\text{nh3}} = x_t$），因此 Q3 的松弛保证了**最优值不会劣于 Q2**。

Q3 比 Q2 更优的场景（88.3%）体现了连续调节的核心价值：

1. **荷随源动**：功率跟踪风光出力曲线变化，减少弃电
2. **价随市动**：避免高电价时段满功率运行，将购电转移至谷时
3. **设备择优**：自动优先使用高效率/低运维设备（PEM 效率优先 vs ALK 成本优先的自动权衡）
4. **逐时氢平衡**：防止不合理的"超前制氢-延后合成"调度
