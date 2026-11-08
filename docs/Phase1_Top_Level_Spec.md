# Phase 1 顶层设计规范与实验演进档案（Frozen Spec & Iterative Audit）

本文档完整记录了 **Phase 1（双臂不透明回弹盒装物机理基准）** 的数学建模、核心算法设计以及四轮迭代实验（v1 $\to$ v4）的归因演进过程。

---

## 1. 核心科学问题与形式化定义

在接触丰富、操作条件易失效（**Transient / Fragile Preconditions**，如带回弹铰链的盒盖、易塌缩的软包袋口）的双臂操作任务中：
- **表面现象**：当前时刻 $t$ 盒盖/袋口处于敞开状态（前置条件 $P_k$ 满足），策略网络据此生成未来 $H$ 步的插入动作块 $A_1 = a_{t:t+H-1}$。
- **核心矛盾**：在动作块执行的中途（$t+\delta$），环境动力学（如盒盖弹簧恢复力矩 $\kappa$ 或袋口柔性回缩）会导致通行净空 $d_{\text{clear}, t+\delta}$ 衰减至安全阈值 $d_{\text{safe}}$ 以下，引发夹手或物块卡死（维护不变量 $I_k$ 失效）。

### 1.1 状态表示与持久化遮挡信念 $b_t$
$$b_t = \big(o_{t-L:t},\; \hat{c}_t,\; \hat{\xi}_t\big)$$
1. **$o_{t-L:t}$**：长度为 $L=12$ 的机器人关节位姿、末端位置与可见物体几何历史（盒盖关闭时盒内物体坐标被遮挡掩码置零）；
2. **$\hat{c}_t \in \{\text{outside}, \text{grasped}, \text{in\_channel}, \text{inside\_box}, \text{uncertain}\}$**：持久化的物体容积状态记忆，解决不透明盒盖闭合后的**目标歧义（Goal Ambiguity）**与**固定窗口遗忘（Execution Memory Loss）**；
3. **$p(\xi \mid b_t)$**：从盒盖近期角度与角速度序列推断的回弹动力学档位后验分布（$\xi \in \{\text{static}, \text{slow}, \text{fast}\}$）。

### 1.2 动作条件因果世界模型 $W_\psi$
为杜绝联合生成模型 $X_\tau = (A_\tau, S_\tau)$ 通过直接修改状态分量 $\nabla_{S_\tau} J(S_\tau)$ 制造“虚假安全预测（State Hallucination）”，未来 $H=24$ 步的条件状态轨迹必须严格作为**干净候选动作终点 $\hat{A}_1$ 的因果函数**：
$$\hat{S}_1(\xi) = W_\psi\big(b_t,\, \hat{A}_1,\, \xi\big), \qquad \hat{s}_{t+\delta} = \big(d_{\text{clear}, t+\delta},\; \phi_{\text{occ}, t+\delta},\; p_{\text{prog}, t+\delta},\; c_{\text{col}, t+\delta}\big)$$

### 1.3 通道门控与终端净空外推风险函数 $\mathcal{R}(\hat{A}_1)$
$$J_{\text{total}}\big(\hat{S}_1(\xi); \hat{A}_1, z_k\big) = \sum_{\delta=1}^{H} \gamma^{\delta} \Big[ \lambda_{\text{clear}} \cdot \phi_{\text{occ}, t+\delta} \cdot \max\big(0,\, d_{\text{safe}} - d_{\text{clear}, t+\delta}\big)^2 + \lambda_{\text{col}} \cdot c_{\text{col}, t+\delta} - \lambda_{\text{prog}} \cdot p_{\text{prog}, t+\delta} + \lambda_{\text{aux}} C_{\text{left\_arm}}(a_{t+\delta}) \Big] + J_{\text{term}}\big(\hat{S}_1(\xi)\big)$$

其中：
- **盒口通道占用门控 $\phi_{\text{occ}, t+\delta} \in [0, 1]$**：仅当物体处于穿越盒口平面的危险区间时激活净空惩罚。若机械臂主动将物体**撤回盒外（$\phi_{\text{occ}} \to 0$）**，即使盒盖完全闭合（$d_{\text{clear}} \to 0$）亦不触发惩罚，从而在数学上允许“先撤回避让 $\to$ 让盒盖落下 $\to$ 重新开盖扶住”的恢复策略自然涌现。
- **终端剩余时间净空外推项 $J_{\text{term}}$**：针对有限预测时域 $H \Delta t = 0.80\text{ s}$ 小于慢速回弹闭合时间（$0.933\text{ s}$）及完整入盒耗时（$\sim 2.6\text{ s}$）引发的“时域短视效应（Horizon Myopia）”，利用预测窗口中后段净空下降速率 $\dot{d}_{\text{close}}$ 与末端剩余未完成进度 $(1 - p_{\text{prog}, t+H})$ 外推完成插入前的最小净空：
  $$\hat{d}_{\text{proj}} = d_{\text{clear}, t+H} - \dot{d}_{\text{close}} \cdot T_{\text{rem}}\big(1 - p_{\text{prog}, t+H}\big), \qquad J_{\text{term}} = \lambda_{\text{term}} \max\big(0,\, d_{\text{warn}} - \hat{d}_{\text{proj}}\big)^2$$

---

## 2. 基于 $\tau_{\text{start}} = 0.7$ 的后阶段混合因果流引导（Late-Stage Hybrid Causal Guidance）

设 Conditional Flow Matching 总去噪采样步数为 $M=10$，去噪时间 $\tau \in [0, 1]$，介入阈值为 $\tau_{\text{start}} = 0.7$：

1. **阶段一：先验模式快速发散期（$\tau \in [0, 0.7)$）**
   不调用世界模型 $W_\psi$，并行积分 $N=8$ 条相关时间噪声粒子，快速凝练多模态动作骨架（涵盖单手直接插 `direct`、双手扶盖插 `hold`、撤回重开 `recover`）：
   $$\frac{d A_\tau^{(n)}}{d\tau} = v_\theta\big(A_\tau^{(n)}, \tau \mid b_t, z_k\big)$$
2. **分水岭离散择优（$\tau = \tau_{\text{start}} = 0.7$）**
   利用一步欧拉终点外推 $\hat{A}_1^{(n)} = A_{0.7}^{(n)} + 0.3\,v_\theta(A_{0.7}^{(n)}, 0.7 \mid b_t, z_k)$，调用一次世界模型计算期望因果风险 $\mathcal{R}(\hat{A}_1^{(n)}) = \mathbb{E}_{\xi \sim p(\xi \mid b_t)}[J_{\text{total}}(W_\psi(b_t, \hat{A}_1^{(n)}, \xi))]$，筛选出最优粒子 $n^\star = \arg\min_n \mathcal{R}(\hat{A}_1^{(n)})$。
3. **阶段二：连续因果梯度精修期（$\tau \in [0.7, 1.0]$）**
   仅对选出的单条最优轨迹在最后 30% 流时间内施加因果终点风险梯度：
   $$\frac{d A_\tau}{d\tau} = v_\theta(A_\tau, \tau \mid b_t, z_k) - \eta(\tau)\, \nabla_{A_\tau} \mathcal{R}\big(\hat{A}_1(A_\tau, \tau)\big)$$
   其中梯度沿因果链 $\nabla_{A_\tau}\mathcal{R} = \big(\frac{\partial \hat{A}_1}{\partial A_\tau}\big)^\top \big(\frac{\partial W_\psi}{\partial \hat{A}_1}\big)^\top \nabla_S J_{\text{total}}$ 反向传播并做范数裁剪。
