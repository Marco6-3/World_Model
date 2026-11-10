# Stage C Flagship Demo Specification: Bimanual Dynamic Bag-Packing (`bag_packing.py`)

请严格按照本规范在 `c:\Users\mingzhe Liu\OneDrive\Desktop\ws\phase1_spring_lid` 中实现并运行论文的压轴旗舰实验 —— **Stage C：双臂动态弹性袋口多物品收纳（Bimanual Dynamic Bag-Packing under Aperture Rebound & Internal Occlusion）**。

---

## 1. 旗舰场景定位与三大科学看点
本实验将我们在 `Stage A (Push-T)` 和 `Stage B (Spring-Lid)` 验证成功的 **Unified 256D Mamba World Model (`Mamba-Past + Mamba-Future`) + Late-Stage In-Flow Causal Guidance ($\tau_{\text{start}} \in [0.7, 0.9]$)** 推向最具实用价值与视觉表现力的高难灵巧操作任务：**将多件异构物品按指定顺序依次收纳进带弹性回缩袋口（Elastic/Collapsible Collar）的立式收纳包中**。

该旗舰任务在一个闭环中同时检验三个核心难点：
1. **弹性袋口非平稳收缩（Transient Aperture Collapse）**：
   收纳包顶部具有弹性收缩袋口（含三种隐刚度/阻尼模态 $\xi \in \{\text{relaxed}, \text{elastic\_slow}, \text{elastic\_fast}\}$）。当左臂松开撑袋口手爪后，袋口双唇向内回弹闭合，右臂必须在有效通行孔径 $d_{\text{aperture}}(t)$ 闭合前完成穿入，或由左臂协同撑开袋口（`hold_collar`）。
2. **包内视觉遮挡与多物品空间记忆（Internal Slot Occlusion & Order Memory）**：
   任务包含两件异构物品（`ItemA`：刚性扁盒；`ItemB`：立式圆柱/方柱），包内底部设有左右两个紧凑收纳槽位（`Slot-Left` 与 `Slot-Right`），且支持两种任务顺序提示（`Order 0: A->B` 先放左槽再放右槽 vs. `Order 1: B->A` 先放右槽再放左槽）。
   - **关键遮挡机制**：一旦第一件物品落入包底槽位（$z < z_{\text{occlude}}$），由于袋身不透明且袋口半合，**包内已放入物品的实时位姿与槽位占用在当前帧观测中被遮挡清零（`occluded = True`）**！
   - 只有具备长程选择性记忆的 **`Mamba-Past`** 能从数十步前的入袋历史中记住“包内哪个槽位已被占用”以及“刚才探测到的袋口回弹刚度 $\xi$”，避免第二件物品砸向已占用槽位造成叠压卡死（Stacking Collision / Jamming）。
3. **中途袋口滑脱紧急撤回与重撑（Mid-Insertion Collar Slip Recovery）**：
   在放入物品途中注入“左臂撑袋滑脱（Collar Slip Perturbation）”——袋口突然高速回缩，規劃器必须基于 `256D Mamba-Future` 预测的孔径净空 $d_{\text{aperture}}$ 骤降，在毫秒内触发 **`withdraw -> re-spread collar -> re-insert`** 安全恢复闭环。

---

## 2. 物理环境与数据设计 (`bag_packing.py`)

基于现有 IsaacLab 5.1 双臂 ALOHA 框架与 RTX 5060 GPU (`C:\isaacsim_env\phase2_cuda`) 实现独立可运行的 `bag_packing.py`：
1. **物理与几何建模**：
   - 在 IsaacLab 场景中构建 `/World/Bag`：包含柔性外观配色袋身、左右双槽底板（`Slot-Left` 中心 $x=-0.048\text{ m}$，`Slot-Right` 中心 $x=+0.048\text{ m}$）以及顶部可动弹性袋口机构（Revolute/Prismatic Spring Collar，由左臂通过拉环/撑口柄 `CollarHandle` 撑开至最大有效孔径 $W_{\text{open}}$，松手后按 $\xi \in \{\text{relaxed}, \text{elastic\_slow}, \text{elastic\_fast}\}$ 动态收缩）；
   - 两件待收纳物品：`ItemA`（尺寸与颜色区分的收纳盒）与 `ItemB`（第二件收纳物），每次回合包含连续两个入袋阶段（`Stage 1: Pack First Item` $\to$ `Stage 2: Pack Second Item under Internal Occlusion` $\to$ `Stage 3: Close Bag Collar`）。
2. **结构化安全状态定义 $\hat{s}_{t+k} \in \mathbb{R}^4$**：
   - $d_{\text{clear}}$：待入袋物品（及右臂手爪）相对收缩袋口边缘的**三维有效通行孔径净空（Effective Aperture Clearance, m）**；
   - $\phi_{\text{occ}} \in [0, 1]$：袋口通道占用门控（物品处于袋口咽喉区时为 1，撤回袋口上方或完全落入包底后归 0）；
   - $p_{\text{prog}} \in [0, 1]$：向目标空闲槽位（正确避开已占用槽位）的归一化入袋进度；
   - $c_{\text{col}} \ge 0$：物品/右臂与收缩袋口唇边或包内已放置物品的非预期接触冲量（$\text{N}\cdot\text{s}$，阈值 $0.004\text{ N}\cdot\text{s}$）。
3. **训练数据采集（Expert + Exploration Rollouts）**：
   - 采集包含 3 种袋口弹性模态（`relaxed`, `elastic_slow`, `elastic_fast`）$\times$ 2 种装包顺序（`A->B`, `B->A`）的专家轨迹（`direct_insert`, `spread_and_hold`, `slip_withdraw_reopen`）以及随机/半落扰动探索轨迹（Sufficient Excitation）。

---

## 3. 模型训练与对比基线（全面使用 $D_z = 256$ `UnifiedMambaWorldModel`）

基于 `Phase 2.1` 中 $D_z = 256$ 取得 `6/6 (100%)` 严格安全排序的结论，在旗舰装包任务上默认采用 **`256D Mamba-Past + 256D Mamba-Future`**（GPU 训练）：
1. **对比方法矩阵（主表 Table C1：常规装包闭环 + 包内遮挡槽位选择）**：
   - `Behavior Cloning (Flow, Short-History H=4)`
   - `Joint-FM (M2, State-Action Joint Generation)`
   - `GPC-RANK (K=16, Post-hoc Ranking at τ=1.0)`
   - `GPC-OPT (K=1, M=20, Post-hoc Single-Shooting Optimization)`
   - `Ours w/ Short-MLP (H=4, 无长程遮挡记忆对照)`
   - `Ours w/ 4D-WM (D_z=4, 无高维动态主干对照)`
   - **`Ours (Full: 256D Mamba-Past/Future + In-Flow Guidance τ_start=0.7 & 0.9)`**
2. **评测指标**：
   - **Dual-Item Packing Success Rate（双物品完整入袋成功率 $\uparrow$）**
   - **Occluded Second-Item Slot Accuracy（遮挡下第二件物品避开已占槽位的正确率 $\uparrow$）**
   - **Bag-Collar & Inter-Item Pinch/Collision Rate（袋口夹碰与叠压碰撞率 $\downarrow$）**
   - **Collar-Slip Perturbed Recovery Rate（中途袋口滑脱紧急撤回重撑恢复率 $\uparrow$）**
   - **Decision Latency P50 (`ms/chunk` $\downarrow$)**
3. **生成可视化轨迹长图 (`outputs/stage_c_bag/bag_packing_demo.png`)**：
   - 自动绘制一张包含 `BC (夹碰/叠压失败)`、`GPC-OPT (超时/碰撞)` 与 **`Ours (Mamba-256D, 成功撑袋入包 + 滑脱撤回重撑 + 遮挡记忆精准分槽)`** 的高分辨率多子图关键帧/三维轨迹与袋口孔径净空演化对比图（`outputs/stage_c_bag/bag_packing_demo.png`），直接用于论文 Figure 1 / Figure 6 展示！

---

## 4. 报告输出要求
实验全部跑通后：
1. 生成详尽的旗舰实验报告 `outputs/StageC_Bag_Packing_实验报告.md`；
2. 将 Stage C 旗舰实验结果汇总进 `outputs/Phase2_GPC_Mamba_实验报告.md` 与 `outputs/evaluation/results.md`，并保存 JSON 数据至 `outputs/stage_c_bag/results.json`。
