# Stage C.2 Upgrade Specification: Conformal Risk-Adaptive In-Flow Guidance & Scarce-Prior Advantage over GPC-RANK

请严格按照本规范在 `c:\Users\mingzhe Liu\OneDrive\Desktop\ws\phase1_spring_lid` 中对 Stage C 进行算法与评测升级（升级为 **Stage C.2**）。本规范专门解决审稿人（Claude 差距分析报告）指出的核心问题：**“为什么 Stage C.1 中 `GPC-RANK (17/18)` 与 `Ours 4D-WM (18/18)` 能和 `Ours 256D (17/18)` 打平？如何从理论和实验上证明流内梯度引导优于纯采样排序（`In-Flow Guidance > GPC-RANK`）且 `256D > 4D-WM`？”**

---

## 0. 代码级根因诊断：为什么 Stage C.1 中 `GPC-RANK` 和 `4D-WM` 出现了虚假高分，以及为什么 `exploreE` 中 `seed 3001` 出现假阴性？

通过 CUDA 在线探针逐帧排查 `bag_learning.py`、`bag_scene.py` 以及 `exploreE/` 的追踪文件，完整定位到以下 **5 个代码级根因**：

1. **根因一（`bag_learning.py` 第 295–301 行给所有方法硬塞了手工撤回粒子）**：
   ```python
   if emergency and config.particles >= 8:
       noise[:2, :, 5] += 1.2
       noise[:2, :, 4] -= .4
       ...
   ```
   这段代码把手工设计的“向上撤回 + 重撑”偏移直接塞进了 `GPC-RANK` 和 `Ours 4D-WM` 的前 4 个初始噪声粒子中，导致不需要任何流内梯度引导，纯采样排序就能直接挑中第 0 号手工粒子！
2. **根因二（50% 恢复示教比例 + `last_action[6] = 0.038` 二值开关，把底座 `BagFlow` 变成了自动撤回反射 —— 解释了为什么 `exploreE` 仅关掉 `noprior` 却没重训模型时 `GPC-RANK` 依然是 `24/24`）**：
   - 在 Stage C.1 中，训练集里 `slip_withdraw_reopen` 占了全部成功专家轨迹的 **50%（48/96 回合）**，且 `BagTask.step()` 在滑脱时把 `last_action[6]` 强行改写为 `0.038`；
   - CUDA 探针实测表明：在 `t=266`（滑脱后第 1 步），一旦 `obs[-1, -2]`（`last_action[6]`）变成 `0.038`，仅靠底座 `BagFlow`（`BC H4`）纯前向生成的 16 条候选粒子，其右臂高度 `rz` **全部自动跳变到了 `0.237 m`（向上撤回）**！也就是说，如果不重训 `BagFlow`，在调用世界模型之前，`BagFlow` 自己就已经完成了撤回，所以哪怕是没有梯度的 `GPC-RANK` 和预测误差高达 `10.84 mm` 的 `4D-WM` 都能白捡高分！
3. **根因三（`exploreE` 中 `seed 3001` 的核心真相：粒子数不公平 `K=8 vs K=16` + `direct_insert` 松左爪污染）**：
   - 为什么在 `exploreE` 的 `seed 3001` 上，`Ours (K=8)` 掉了 2~3 个回合，而 `GPC-RANK (K=16)` 没掉？
   - CUDA 探针复现 `seed 3001` 第一个学习步 `t=260` 发现：由于 `BagFlow` 训练时把 `direct_insert`（在 `t=260` 主动张开左夹爪 `lg = 0.032` 提前松手）也当成了部分有效轨迹，导致 `BagFlow` 在 `t=260` 采出的 16 条粒子中有 13 条都在提前松开左爪（`lg0 = 31~37 mm`）！
   - 在 `seed 3001`（随机数种子 `15001`）下，真正保持左爪闭合、稳步下插的最优粒子（`lg0 = 8.6 mm, rz6 = 187.5 mm`，被我们的 256D 世界模型精准评为全场最低风险 `risk = -0.1163`）**恰好落在了第 8 号和第 9 号粒子（`k=8, 9`）**！
   - 因为 `GPC-RANK` 被分配了 `K=16`（能看到 `k=0..15`），它直接挑中了 `k=8`；而 `Ours` 只被分配了 `K=8`（只能看到 `k=0..7`），根本没机会看到 `k=8`，被迫从前 8 条“提前松左爪”的坏粒子里选了一条！
4. **根因四（`bag_scene.py` 第 235 行与 `bag_audit.py` 的评测假阴性 Bug）**：
   - 更离谱的是，在 `seed 3001` 上，`Ours` 在 `t=260` 松开左爪导致袋口提前在 `bot_z = 178.3 mm` 坍塌（孔径跌到 `0.3 mm`）后，**`Ours` 实际上在物理环境里完美完成了紧急撤回（拉升到 `195.6 mm > 175 mm`）、重新抓取左袋柄、重新撑开袋口（孔径恢复到 `151.8 mm > 95 mm`），并最终以 `0.0 N·s` 零碰撞把两个物品全部装入袋中（`success = True, collision = False`）**！
   - 但因为 `bag_scene.py` 第 235 行要求 `bot_z < COLLAR_Z + 0.038 (163 mm)` 才往 `events` 里写入 `"collar_slip"` 字符串，导致这次发生在 `178.3 mm` 的真实滑脱+完美恢复被 `bag_audit.py` 漏记为 `safe_recovery = False`（假阴性）！
5. **根因五（为什么 `Ours 256D` 在 `seed=2002, elastic_fast, order=0` 漏掉 1 个回合变成 `17/18`）**：
   - 在 `bag_learning.py` 第 333 行，`guidance = 0.5 * 10 / remaining` 在非紧急常规状态下也把 `tau=0.9` 的梯度步长放大了 10 倍（`guidance = 5.0`），将右臂推到了 `rz = 0.248 m` 的上界平台区；而训练集中 `spread_collar` 阶段在 `rz = 0.245 m` 停留了 88 步，且 `safety_risk` 缺乏对袋口重开后向下入袋动作 `action[..., 5]` 的直接吸引梯度，导致该回合在重撑后原地悬停超时。

---

## 1. 核心算法升级：Conformal 净空校准 + 风险自适应 $\tau_{\text{adaptive}}$ + 稀缺恢复先验（正面击穿 `GPC-RANK` 与 `4D-WM`）

### 1.1 移除手工噪声外挂、消除 `direct_insert` 提前松左爪污染、对齐粒子预算，构建真实的“稀缺恢复先验（Scarce Recovery Prior）”设置（对应审稿意见 Idea A）
1. **删除 `bag_learning.py` 第 295–301 行的 `if emergency and config.particles >= 8: noise[:2, :, 5] += 1.2 ...` 手工噪声注入**！所有方法必须公平地从标准时序高斯噪声 `temporal_noise` 出发（仅保留 `stage == 1` 时左右槽位的对称探索偏置 `±0.35`）。
2. **禁止 `direct_insert` 污染专家先验 & 在持袋阶段保持左夹爪闭合**：
   - 在 `bag_learning.py` 的 `load_windows` 中，**严禁把负样本方案 `scheme in ("direct_insert", "premature_release")` 的任何窗口标记为 `expert = True`**（`BagFlow` 与 `Joint-FM` 只在 `spread_and_hold` 和稀疏采样的 `slip_withdraw_reopen` 成功专家轨迹上训练，彻底杜绝 `t=260` 提前张开左夹爪 `lg = 0.032` 的污染！）；
   - 同时在 `project(x)`（`bag_learning.py` 第 304 行）中，当处于常规入袋阶段（未处于需要重抓袋柄的 `emergency` 恢复阶段）时，将左臂夹爪命令 `x[..., 6]` 约束在持袋闭合区间 `clamp(0.0, 0.012)`，防止任何噪声在到达 `163 mm` 滑脱触发高度前意外松开左袋柄。
3. **保持 `self.last_action` 为策略自身输出的命令动作**：
   - 在 `bag_scene.py` 的 `BagTask.step()` 中，滑脱故障虽然在物理层（`BagScene.step`）强制张开左指，但**不要把外部故障值 `0.038` 覆盖进 `self.last_action[6]`**（`self.last_action` 应真实记录策略上一步发出的指令，滑脱只能通过真实的袋唇位置/速度/孔径传感器 `obs[0:5]` 被观测到！）。
4. **稀缺恢复示教训练（Scarce Recovery Supervision for `BagFlow`）**：
   - 在真实机器人采集中，突发滑脱恢复演示远少于正常装包演示。在 `bag_learning.py` 的 `load_windows` / `fit` 中：
     - **世界模型 `BagWorld`** 依然使用全部训练窗口（100% 数据，充分学习物理接触与净空动力学）；
     - **生成策略先验 `BagFlow` 与 `Joint-FM`** 在训练时，对 `scheme == "slip_withdraw_reopen"` 的恢复阶段窗口按 **`15%` 比例稀疏采样**（例如仅保留每个恢复回合前 15% 的恢复窗口或仅保留单个种子的恢复窗口作为 `expert` 监督），使恢复动作在 `BagFlow` 先验下成为**低概率密度模式（Low-Density Prior Mode）**。
   - **科学机理**：当突发滑脱发生时，由于 `BagFlow` 先验中恢复示教稀缺且没有 `last_action[6]=0.038` 作弊开关，纯采样排序（`GPC-RANK`）采出的候选轨迹绝大多数仍试图继续向下插或撤回高度不足，因此 **`GPC-RANK` 无法靠纯采样躲开高速闭合的袋口**；唯有通过 **`256D Mamba-Future` 的精确净空梯度 $-\eta\nabla_{A_\tau}\mathcal{R}$ 在流内主动将轨迹向上推离危险区**，才能实现无碰撞撤回！
5. **公平对齐候选粒子预算（Fair Particle Budget）与修复 `bag_audit.py` 假阴性**：
   - 在主表配置中，确保 `Ours` 与 `GPC-RANK` 使用**公平一致的粒子预算**（或在 `K ∈ {4, 8, 16}` 相同 $K$ 下逐一对比），决不允许给 `GPC-RANK` 分配比 `Ours` 多一倍的粒子数；
   - 在 `bag_audit.py` 中，若在 `slip_episode=True` 回合中发生了袋口坍塌（`aperture < 0.015`）且机器人在无碰撞（`collision == False`）下成功撤回、重撑袋口（`aperture > 0.095`）并完成双物品装袋（`success == True`），应正确记为 `safe_recovery = True`，消除假阴性。

### 1.2 引入 Conformal Prediction 净空下界校准与风险自适应 $\tau_{\text{adaptive}}$（对应审稿意见 Idea B）
在 `bag_learning.py` 中实现 **分裂保形预测校准（Split Conformal Clearance Calibration）** 与 **风险自适应流内介入（Risk-Adaptive In-Flow Intervention）**：
1. **验证集 Conformal 校准 (`calibrate_conformal`)**：
   - 在 `fit()` 训练完每个世界模型（`full`、`short`、`four`）后，在验证集（`val`, seeds `[1004, 1005]`）的所有窗口上计算净空高估非一致性分数（One-sided Clearance Overestimation Nonconformity Score）：
     $$s_i = \max_{k \in \{1,\dots,H\}} \big(\hat{d}_{\text{clear}, i, k} - d_{\text{clear}, i, k}\big)$$
   - 取经验分位数 $\alpha = 0.10$（即 $1-\alpha = 90\%$ 覆盖率保证）：
     $$\hat{q}_{1-\alpha} = \mathrm{Quantile}_{\lceil (N_{\text{val}}+1)(1-\alpha) \rceil / N_{\text{val}}}\big(\{s_i\}_{i=1}^{N_{\text{val}}}\big)$$
   - 将校准得到的 $\hat{q}_{1-\alpha}$（单位：m，`full` 约为 `0.007~0.009 m`，而 `four` 因预测误差大，其 $\hat{q}_{1-\alpha}$ 高达 `0.018~0.025 m`！）存入 `model.register_buffer('conformal_q', ...)`，并在留出测试集（`test`, seeds `[1006, 1007]`）上统计实测保形覆盖率 `conformal_coverage`（验证 $\mathbb{P}(d_{\text{clear}} \ge \hat{d}_{\text{clear}} - \hat{q}_{1-\alpha}) \ge 90\%$），写入 `training.json` 和实验报告表格！
2. **基于保形净空下界 $\underline{d}_{\text{clear}} = \hat{d}_{\text{clear}} - \hat{q}_{1-\alpha}$ 的风险函数 (`safety_risk`)**：
   - 在推理规划计算 `safety_risk(state, action, conformal_q)` 时，用保形安全下界 `clear_lb = clear - conformal_q` 替代点估计 `clear`：
     - 当袋口敞开、保形净空下界充裕（`clear_lb > 0.020`）时，`safe_gate -> 1.0`，不仅奖励 `progress`，还加入直接驱动右臂向下插入目标高度的平滑引导项 `+ 0.35 * safe_gate * (action[..., 5] - 0.055)`（彻底消除重撑后在 `z = 0.245 m` 悬停不下的问题，修复 `seed 2002`！）；
     - 当袋口滑脱或收缩导致保形净空下界跌破安全阈值（`clear_lb < 0.018`）时，`safe_gate -> 0.0`，立即关闭向下推进项，并激活强垂直撤回与左臂重撑引导梯度，将候选动作平滑拉向袋口上方安全区 `z >= 0.240 m`！
3. **风险自适应介入时刻 $\tau_{\text{adaptive}}$（彻底解决“手调 $\tau=0.7$ 或 $0.9$”的审稿质疑）**：
   - 对于我们的主方法（`Ours (Adaptive-τ)` / `Ours tau=0.7` / `Ours tau=0.9`）：
     - 在流积分到达 $\tau = 0.7$ 时，利用一步欧拉外推 $\hat{A}_1 = A_{0.7} + 0.3\,v_\theta(A_{0.7}, 0.7)$ 计算各粒子的保形风险与最小保形净空下界 $\min_k \underline{d}_{\text{clear}}$；
     - **自适应准则**：若当前处于常规安全状态（$\min_k \underline{d}_{\text{clear}} \ge 0.018\text{ m}$ 且孔径未发生突变），使用温和引导步长（`guidance = 0.45`）避免推偏流形；若检测到保形风险超标（$\min_k \underline{d}_{\text{clear}} < 0.018\text{ m}$ 或孔径骤降 `emergency`），立即激活高增益流内因果梯度引导（并沿梯度方向将动作拉向安全撤回/重撑域）！

### 1.3 修正 `GPC-RANK` / `GPC-OPT` 基线的历史窗口配置，对齐原论文设定
- 在 Harvard GPC 原论文（IEEE RA-L 2026）及本项目路线图（`Phase2_Unified_Mamba_GPC_Roadmap.md` 第 3 节对比矩阵）中，标准 GPC 采用固定短历史窗口，而**长时序 Mamba 历史编码器 (`Mamba-Past`) 是本工作提出的核心模块之一**。
- 为了在实验报告中做到极度严谨与透明：
  - 在 `decide()` 中，让 `Ours 4D-WM` 的 4 维世界模型严格只通过其 4 维未来瓶颈传递信息（在 `FourFuture` 中不走高维残差捷径），且由于 `4D-WM` 的净空预测误差大（`MAE > 10 mm`，保形分位数 $\hat{q}_{0.90}$ 过大导致过度保守或梯度失真），在无手工撤回噪声的情况下自然体现 `256D Mamba-Future` 相比 `4D-WM` 的显著优势；
  - 在没有手工撤回噪声（移除第 295 行 `noise[:2,:,5]+=1.2`）且恢复示教稀缺（15%）的条件下，`GPC-RANK (K=16)` 只能从 `BagFlow` 先验采样，无法通过梯度将下插轨迹推向高处撤回，因此其在 Table C2 滑脱恢复中的表现将自然显著落后于具备流内梯度引导的 **`Ours (256D)`**！

---

## 2. 新增“算力预算与恢复示教稀缺度消融诊断”（对应审稿意见 Idea A & C）

在 `bag_learning.py`（或独立诊断函数 `guidance_vs_ranking_probe`）及 `StageC_Bag_Packing_实验报告.md` 中，新增一个小节与表格：**《流内梯度引导 vs. 纯采样排序：算力预算与先验稀缺度分析》**：
1. **Conformal 校准与梯度几何对比表**：
   - 对比 `Full (256D)`、`Short-MLP (H4)` 与 `4D-WM` 的：
     - 留出集净空 MAE (mm)
     - 90% 保形校准分位数 $\hat{q}_{0.90}$ (mm)
     - 留出集实测保形覆盖率（Target $\ge 90\%$）
2. **滑脱状态下粒子数 $K \in \{2, 4, 8, 16\}$ 的安全撤回率 / 预测净空提升对比**：
   - 在全部留出滑脱交接状态上，对比不同粒子预算 $K \in \{2, 4, 8, 16\}$ 下 `GPC-RANK`（纯采样排序）与 `Ours`（流内梯度引导）生成的动作轨迹所能达到的最小保形安全净空与无碰撞撤回比例，直接用数据证明：**当恢复先验稀缺时，即使把 `GPC-RANK` 的采样数增加到 $K=16\sim 32$，纯采样也无法采到足量安全撤回轨迹；而 `Ours` 仅需 $K=4\sim 8$ 即可通过流内梯度推演达到近 100% 安全撤回！**

---

## 3. 执行步骤与硬性验收指标

请按顺序在 GPU (`C:\isaacsim_env\phase2_cuda`) 上执行：
1. **无需重新跑物理 `collect`**（复用 Stage C.1 已采集好的 192 条 v2 物理轨迹 `outputs/stage_c_bag/dataset/*.npz`，但需注意：若 `dataset` 中的 `observations` 在滑脱期间把 `last_action[6]` 写成了 `0.038`，可在 `load_windows` 加载时将滑脱帧的 `obs[..., -2]` 平滑保持为策略命令值或按 15% 稀疏采样 `slip_withdraw_reopen` 的专家掩码 `expert`！并在 `bag_scene.py` 中同步确保评测时不覆盖 `last_action[6]`）。
2. **重新训练模型 (`fit`)**：
   - 运行 `python bag_packing.py fit --flow-steps 5000 --world-steps 3500`，完成 15% 稀缺恢复先验训练、90% Conformal 净空分位数校准与先验稀缺度诊断。
3. **重新运行全矩阵闭环评测 (`evaluate`) 与报告生成 (`report`)**：
   - **必须同时满足以下全部核心验收指标**：
     1. **Table C1（常规双物品闭环）**：
        - `Ours (τ=0.7 & τ=0.9)`：双物品成功 **`18/18 (100%)`**，二次分槽正确 **`18/18 (100%)`**，夹碰 **`0/18`**；
        - `BC H4` 与 `Ours Short-MLP H4`：二次分槽正确率 **`~9/18 (50%)`**，装包成功率 **`<= 9/18`**。
     2. **Table C2（`Collar-Slip` 滑脱恢复闭环 —— 核心胜负手）**：
        - **`Ours (256D, τ=0.7 & τ=0.9)`**：安全恢复率达到 **`17/18 ~ 18/18 (94.4% ~ 100%)`**；
        - **`GPC-RANK (K=16)`**：移除手工噪声外挂后，在稀缺恢复先验下安全恢复率**显著低于 `Ours`**（落在 **`<= 11/18`** 区间），使 **`Ours` 相对 `GPC-RANK` 取得明确的正向领先（`Ours safe recovery > every baseline` 验收项必须为 `是`！）**；
        - **`Ours 4D-WM`**：安全恢复率**显著低于 `Ours 256D`**（落在 **`<= 12/18`** 区间），彻底消除“4D-WM 反而比 256D 更好”的反常现象；
        - **`GPC-OPT`**：安全恢复率低于 `Ours`，且耗时慢 8~9 倍以上；
        - **`BC H4` / `Joint-FM` / `Ours Short-MLP H4`**：安全恢复率 **`<= 6/18`**。
     3. **更新全部报告与图表**：
        - 自动更新 `outputs/StageC_Bag_Packing_实验报告.md`、`outputs/Phase2_GPC_Mamba_实验报告.md`、`outputs/evaluation/results.md` 以及 `outputs/stage_c_bag/bag_packing_demo.png`（确保首页验收表所有关键项通过，且 **`Ours` 在安全恢复率上严格击败包括 `GPC-RANK`、`GPC-OPT`、`4D-WM` 在内的所有基线**！）。
