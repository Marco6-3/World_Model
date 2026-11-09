# Phase 2 Stage A/B Codex Execution Specification

请严格按照本规范在 `c:\Users\mingzhe Liu\OneDrive\Desktop\ws\phase1_spring_lid` 中执行代码升级与实验。只做架构升级、公开数据集（Push-T）基准测试、GPC 基线对比与 Mamba 升维消融实验，直接跑通全流程并生成实验报告。

---

## 0. 保护已有 v4 结果快照
在修改代码和运行新实验前，先将当前的 `v4` 报告备份：
- 复制 `outputs/Phase1_实验报告.md` 为 `outputs/Phase1_实验报告_v4.md`
- 复制 `outputs/evaluation/results.md` 为 `outputs/evaluation/results_v4.md`

---

## 1. 核心架构升级：统一 128 维双向 Mamba 世界模型与两层解耦读出头 (`learn.py` / `mamba_wm.py`)

为避免 Windows 下编译 Linux 专属 `causal-conv1d` / `mamba-ssm` C++ 扩展失败，请使用 **纯 PyTorch 原生算子** 实现标准 **Selective State Space Model (`MambaBlock`)**：
1. **`MambaBlock(d_model=128, d_state=16, d_conv=4, expand=2)`**：
   - 输入投影：将输入 $u_t \in \mathbb{R}^{D}$ 映射至主分支 $x_t \in \mathbb{R}^{E}$ 与门控分支 $z_t \in \mathbb{R}^{E}$（$E = \text{expand} \cdot D$）；
   - 因果局部卷积：`nn.Conv1d(E, E, kernel_size=d_conv, groups=E, padding=d_conv-1)` 截断至序列长度 $L$ 后接 `SiLU`；
   - 选择性状态空间参数（Input-Dependent Selection）：
     $$\Delta_t = \mathrm{softplus}(W_\Delta x_t + b_\Delta) \in \mathbb{R}^{E}, \quad B_t = W_B x_t \in \mathbb{R}^{N_{\text{state}}}, \quad C_t = W_C x_t \in \mathbb{R}^{N_{\text{state}}}$$
   - 零阶保持（ZOH）离散化与选择性扫描（Selective Scan）：
     对角状态矩阵 $A = -\exp(\text{A\_log}) \in \mathbb{R}^{E \times N_{\text{state}}}$，$\bar{A}_t = \exp(\Delta_t \odot A)$，$\bar{B}_t = \Delta_t \odot B_t$；
     按时间步递推隐状态 $h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t \odot x_t$，输出 $y_t = (h_t \cdot C_t) + D \odot x_t$，再乘以门控 $\mathrm{SiLU}(z_t)$ 并经线性层映射回 $\mathbb{R}^{D}$ + 残差连接 + `LayerNorm`。

2. **双向时序 Mamba 世界模型 (`UnifiedMambaWorldModel`)**：
   - **向后看（`Mamba-Past` 历史与遮挡编码器）**：
     输入历史观测序列 $o_{t-L+1:t}$，经 `MambaBlock` 扫描输出高维信念态 $b_t \in \mathbb{R}^{D_z}$（默认 $D_z = 128$），并由轻量分类头 $q_\phi(\xi \mid b_t)$ 预测隐动态模态分布；
   - **向前看（`Mamba-Future` 高维动态推演 Tier-1 + 可插拔读出头 Tier-2）**：
     - **Tier-1（通用高维状态空间递推）**：以 $\hat{z}_t = \phi_{\text{init}}(b_t, \xi) \in \mathbb{R}^{D_z}$ 为起点，将未来候选动作块 $\hat{A}_1 = (\hat{a}_t, \dots, \hat{a}_{t+H-1})$ 逐帧与状态融合送入 `Mamba-Future` 递推得到高维轨迹 $\hat{z}_{t+1:t+H} \in \mathbb{R}^{H \times D_z}$；
     - **Tier-2（双模式可插拔读出头）**：
       - `g_state(z)`（通用状态重构头）：预测未来完整环境/物体状态 $\hat{x}_{t+k}$（在公开数据集 Push-T 上直接预测 T 块位姿 $(x, y, \theta)$、速度与末端坐标及目标对齐误差；在弹簧盖任务上辅助重构盒体/物体几何状态以提供密集自监督）；
       - `g_safe(z)`（结构化安全读出头）：在弹簧盖等瞬态受限任务上从 $\hat{z}_{t+k} \in \mathbb{R}^{128}$ 解码出 $[\hat{d}_{\text{clear}}, \hat{\phi}_{\text{occ}}, \hat{p}_{\text{prog}}, \hat{c}_{\text{col}}]^\top$。

---

## 2. 新增 Harvard GPC (IEEE RA-L 2026) 基准算法对比 (`inference.py`)

在推理规划器中完整实现 Harvard GPC 的两种核心模式，与我们的 $\tau_{\text{start}}=0.7$ 流内因果引导同台对比：
1. **`GPC-RANK`（纯后验采样重排序，对应 $\tau_{\text{start}}=1.0, M=0$）**：
   先用 Flow 策略完整积分到 $\tau=1.0$ 生成 $K$ 条候选动作序列 $\{A_1^{(k)}\}_{k=1}^K$，送入世界模型评估累积风险/奖励 $\mathcal{R}(W(b_t, A_1^{(k)}))$，返回最优序列 $A_1^{(k^\star)}$。
2. **`GPC-OPT`（生成后流形外单打靶梯度优化，对应 $\tau=1.0$ Warm-Start + $M$ 步后验梯度下降）**：
   先完整生成动作序列 $\hat{A}^{(0)} = A_1 \sim \text{Flow}(o_t)$，随后在流匹配生成流形**之外**直接对动作变量执行 $M=15\sim 25$ 步梯度优化：
   $$\hat{A}^{(\ell)} = \mathrm{Proj}_{\mathcal{A}}\Big(\hat{A}^{(\ell-1)} - \eta_{\text{opt}} \nabla_{\hat{A}} \mathcal{R}\big(W(b_t, \hat{A}^{(\ell-1)})\big)\Big), \quad \ell = 1, \dots, M$$
3. **`GPC-RANK+OPT`**：对 $K$ 条后验样本分别执行 `GPC-OPT` 后选最优。
4. **`Ours (Unified Mamba-WM + Late-Stage In-Flow Guidance, τ_start=0.7)`**：
   - $\tau \in [0, 0.7)$：仅运行纯 Flow 速度场积分（0 世界模型开销）；
   - $\tau = 0.7$：一步欧拉终点映射 $\hat{A}_1^{(k)} = A_\tau^{(k)} + (1-\tau)v_\theta$，通过 `128D Mamba-Future` 筛选最优模态粒子；
   - $\tau \in [0.7, 1.0]$：在流匹配最后 $30\%$ 积分步内沿流形注入因果梯度 $-\eta(\tau)\nabla_{A_\tau}\mathcal{R}(\hat{A}_1)$。

---

## 3. 实验一（Stage A）：公开基准 Push-T 正面对标 Harvard GPC (`pusht_benchmark.py`)

创建一个独立的公开基准评测脚本 `pusht_benchmark.py`：
1. **环境与数据准备**：
   - 若环境中未安装 `pymunk` / `shapely` / `zarr`，可通过 `C:\isaacsim_env\Scripts\python.exe -m pip install pymunk shapely zarr huggingface_hub` 安装；
   - 支持自动拉取公开 Push-T 状态数据集（如 HuggingFace `lerobot/pusht` parquet/zarr 或 Columbia 官方 `pusht_cchi_v7_replay.zarr.zip`），并在无外网/下载受限时内置基于 `pymunk`（或精确 2D 刚体接触动力学）的标准 **Push-T 平面推 T 块环境**，采集包含 **Expert Demonstrations + Random Perturbation Rollouts（与 GPC 论文 §III "sufficient excitation" 完全一致）** 的标准化数据集。
2. **评测指标（复刻并扩展 GPC 论文 Table I & Table III）**：
   - 在相同测试种子集（$\ge 50$ seeds）上对比：
     1. `Behavior Cloning (Flow Policy, K=1, M=0)`
     2. `GPC-RANK (K=16, M=0)` 与 `GPC-RANK (K=64, M=0)`
     3. `GPC-OPT (K=1, M=20, Post-hoc)`
     4. `GPC-RANK+OPT (K=8, M=15, Post-hoc)`
     5. `Early In-Flow Guidance (τ_start=0.0)`
     6. **`Ours: Mamba-WM (128D) + Late-Stage In-Flow (τ_start=0.7)`**
   - 记录：**Final Target Coverage / Alignment Score (IoU / Pose Score $\uparrow$)**、**Success Rate ($\uparrow$)**、**Adjacent Action Smoothness Jump ($\downarrow$)**、**World Model Rollout Calls ($\downarrow$)** 与 **Wall-Clock Decision Latency (`ms/chunk` $\downarrow$)**。

---

## 4. 实验二（Stage B）：Mamba 归纳偏置、维度扩展 ($D_z$) 与弹簧盖全闭环消融

在 `phase1_spring_lid` 弹簧盖瞬态闭合环境上运行三组关键消融并升级主表：
1. **消融 1 —— 历史编码器序列归纳偏置（长历史 + 遮挡记忆测试）**：
   构造 **遮挡/延迟交接测试（Occluded-History Regime Identification）**：在交接前有 $L_{\text{hist}} = 24$ 步历史，但仅前 6 步存在盖子运动观测，后 18 步将盖子角速度观测マスク（模拟视觉遮挡或长等待窗口），对比：
   - `Short-Window MLP (H=4)`（因窗口截断丢失早期动态，准确率退化）
   - `Full-Window MLP (H=24)`（参数过拟合噪声）
   - `Recurrent GRU (H=24)`
   - **`Mamba-Past (Selective SSM, D_z=128, H=24)`**（验证选择性门控记住早期瞬态脉冲并维持高辨识准确率与低轨迹误差）。
2. **消融 2 —— 世界模型表征维度消融 ($D_z \in \{4, 32, 128, 256\}$)**：
   对比直接在 4 维手工状态空间递推 vs. 升维至 $32 / 128 / 256$ 维 `Mamba-Future` + 结构化安全读出头在未来状态预测误差、碰撞漏报率（False Negative Rate）与闭环成功率上的差异。
3. **消融 3 —— Phase 1.5 弹簧盖主表升级（含 `GPC-RANK` / `GPC-OPT` 对比与自适应短步长 `COMMIT`）**：
   - 将非饱和预警净空阈值校准在合理区间（避免宽净空梯度饱和），并在检测到高速回弹（$\hat{p}(\xi=\text{fast}) > 0.4$ 或 $\hat{\omega}_{\text{lid}}$ 较高）时启用自适应短执行窗口（`COMMIT = 4`），消除 4 步预释放带来的重接触冲量，使 `Ours (Mamba-128D, τ_start=0.7)` 在离线排序、常规闭环（`static`/`slow`/`fast`）与半落扰动恢复（Perturbed Recovery）上全面击败 `BC`、`Joint-FM (M2)`、`GPC-RANK` 与 `GPC-OPT`。

---

## 5. 报告输出要求
实验全部跑通后，自动生成/更新以下文件：
1. `outputs/Phase2_GPC_Mamba_实验报告.md`（详尽中文实验报告，包含 Push-T 公开基准对标 GPC 表、Mamba vs. GRU vs. MLP 遮挡记忆消融表、$D_z$ 维度扩展消融表、$\tau_{\text{start}}$ 帕累托曲线表、以及弹簧盖闭环与扰动恢复对比表）；
2. 同步更新 `outputs/Phase1_实验报告.md` 与 `outputs/evaluation/results.md`，确保所有表格数据真实来自最新评测 JSON。
