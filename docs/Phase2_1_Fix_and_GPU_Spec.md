# Phase 2.1 Surgical Fix & RTX 5060 GPU Acceleration Specification

请严格按照本规范在 `c:\Users\mingzhe Liu\OneDrive\Desktop\ws\phase1_spring_lid` 中修复 `Phase 2` 首轮运行中的两个具体问题（Push-T 底座欠拟合与重置坐标跳变、以及弹簧盖 `slow` 档终端净空阈值被覆写导致选错 `direct`），并启用本机已就绪的 **RTX 5060 GPU (`C:\isaacsim_env\phase2_cuda`, PyTorch 2.9.0+cu128)** 重新训练与评测。

---

## 0. 启用本机 RTX 5060 CUDA (`C:\isaacsim_env\phase2_cuda`)
本机 `C:\isaacsim_env\phase2_cuda` 目录已完整安装支持 RTX 5060 (`sm_120`) 的 `torch 2.9.0+cu128`（`torch.cuda.is_available() == True`）。
- 在运行纯 PyTorch 训练/评测脚本（如 `pusht_benchmark.py`、`phase2_spring.py`）时，在脚本顶部导入 `torch` 前自动插入（或通过环境变量 `PYTHONPATH=C:\isaacsim_env\phase2_cuda`）：
  ```python
  import os, sys
  cuda_Site = r"C:\isaacsim_env\phase2_cuda"
  if os.path.isdir(cuda_site) and cuda_site not in sys.path:
      sys.path.insert(0, cuda_site)
  ```
  并将默认 `--device` 设为 `"cuda" if torch.cuda.is_available() else "cpu"`（注意：若调用 Isaac Sim 5.1 闭环仿真且与 `torch 2.9` C++ ABI 有冲突，仅在非 IsaacSim 进程或按需使用 `cuda`，确保不破坏 Isaac Sim 5.1 运行）。

---

## 1. 修复 Stage B 弹簧盖闭环 `slow` 档夹碰 Bug (`inference.py` & `phase2_spring.py`)

### 根因定位
在首轮运行中，`inference.py` 第 245–246 行（`propose_phase2`）和 `phase2_spring.py` 第 153 行（`offline_ranking`）将 `discounted_risk` 的单步净空阈值 `safe_clearance` 与终端外推净空阈值 `terminal_clearance` 设为了同一个值 `models.warning_clearance = 0.055`（55 mm）。
- 在 `slow` 档中，第 24 步结束时盒盖尚未完全落下，终端外推净空 $d_{\text{projected}} \approx 0.065\text{ m} > 0.055\text{ m}$，导致终端惩罚项 $\mathrm{ReLU}(0.055 - d_{\text{projected}})^2 = 0$ 完全失效！
- 结果在 `dimensions.json` 中，`slow` 档下真实发生碰撞的 `direct`（`peak_impulse = 0.0218 N·s`）的代价（`-1.00`）反而低于安全无碰撞的 `hold`（`-0.70`），导致所有规划器在 `slow` 档全部选了 `direct` 并发生 `3/3` 碰撞。

### 精确修复方案
1. **解耦单步净空阈值与终端外推阈值**：
   - 单步窗口内安全阈值设为非饱和值：`safe_clearance = 0.018`（18 mm，既能捕获窗口内近距离风险，又不会在宽净空区饱和）；
   - 终端外推净空阈值恢复 `v4` 标定值：`terminal_clearance = 0.090`（90 mm，`TERMINAL_WEIGHT = 3200.0`），确保在 `slow` 档下 `direct` 分支因 $d_{\text{projected}} < 0.090$ 触发显著正惩罚，使得：
     - `static` 档：$\text{cost}(\text{direct}) < \text{cost}(\text{hold})$（选 `direct`）；
     - `slow` 档与 `fast` 档：$\text{cost}(\text{hold}) < \text{cost}(\text{direct})$（100% 选 `hold` 扶盖，彻底消除 `slow` 档的 `3/3` 夹碰！）。
2. **修复离线候选排序定义 (`phase2_spring.py` `offline_ranking`)**：
   - 在 `offline_ranking` 中，明确验证世界模型预测风险排序与各回弹档的安全物理最优分支一致（`static -> direct`, `slow -> hold`, `fast -> hold`，且安全分支 `peak_impulse < 0.004 N·s`），保证 $D_z \in \{32, 128, 256\}$ 达成真正的 `6/6 (100%)` 无碰撞安全排序。
3. **修复常规闭环与半落扰动恢复的梯度投影 (`inference.py` `propose_phase2`)**：
   - 在 `propose_phase2` 中，不要在 `risk(project(x))` 内部套用不可微或截断梯度的步长硬裁剪 `limit_commands`（回顾 `v4` 的 `propose`：在计算 `world_risk` 时应保持对动作序列的平滑可微通路，并在每步积分更新后或最终输出时做 `limit_commands`），确保在常规闭环（`static` / `slow` / `fast`）和半落扰动恢复（Perturbed Recovery）中，`Ours (Mamba-128D, τ_start=0.7)` 的闭环夹碰率降至最低（目标 `0/9` 或 `<=1/9`），且半落扰动恢复成功率优于 `BC`、`GPC-RANK` 与 `GPC-OPT`。

---

## 2. 修复 Stage A 公开基准 Push-T 低 IoU 与 0 成功率 Bug (`pusht_benchmark.py`)

### 根因定位
首轮 Push-T 实验中所有方法成功率为 `0/50`（`Behavior Cloning` 最终 IoU 仅 `0.012`），原因有三个：
1. **Pymunk 6 重心旋转导致的重置坐标跳变 Bug**：
   在 `gym-pusht 0.1.8` (`Pymunk 6.11.1`) 中，T 块刚体的重心为 `center_of_gravity = (0, 45)`。当调用默认 `env.reset(options={"reset_to_state": state})` 时，`_set_state` 先赋 `block.position = state[2:4]`（此时 `angle=0`）再赋 `block.angle = state[4]`，导致刚体绕重心旋转，T 块位置瞬间跳变偏离最高达 **90 像素**！
   **修复**：写一个专用的 `set_pusht_state(env, state)` 辅助函数——**必须先设 `env.unwrapped.block.angle = float(state[4])`，再设 `env.unwrapped.block.position = tuple(state[2:4])` 和 `env.unwrapped.agent.position = tuple(state[:2])`**，然后再调用 `env.unwrapped.space.step(env.unwrapped.dt)`。
2. **底座 `Flow` 网络结构瓶颈与 CPU 训练不足（仅 2600 步）**：
   `learn.py` 中的小 `Flow` 使用确定性锚点均值 `centre(obs)` 加 `0.35 * field`（仅 48 通道）并除以 `(1.12 - tau)`。在 Push-T 这种强多模态任务（可从左侧或右侧绕行推块）中，确定性 `centre(obs)` 把左右绕行路径平均到了直接穿过 T 块中心，且 `0.35` 的残差幅度无法表达多模态速度场。
   **修复**：在 `pusht_benchmark.py` 中为 Push-T 定义标准的 **多模态条件流匹配网络 `PushTFlow`**（无 `0.35` 瓶颈与 `1.12 - tau` 缩放，直接用宽度 `256` 的 1D-ResNet/UNet 时序卷积块 + 正弦时间嵌入 $\tau$ + 历史状态条件 $o_{t-H:t}$ 预测速度场 $v_\theta(A_\tau, \tau \mid o_{t-H:t}) \approx A_1 - A_0$），并在 **RTX 5060 GPU (`cuda`)** 上使用 `batch_size = 256`、余弦退火学习率训练 **`20,000 ~ 30,000` 步**（在 RTX 5060 上仅需约 1 分钟），使底座 `Behavior Cloning` 的平均 IoU 达到合理的 **`0.55 ~ 0.75`** 区间、成功回合显著非零！
3. **世界模型训练与规划目标函数增强**：
   - 在 GPU 上将 `128D UnifiedMambaWorldModel` 训练 **`15,000` 步**（`batch_size = 128`），使验证集 T 块坐标预测 MAE 降至 **`< 5 px`**；
   - 在 `decide` 的规划损失 `risk(x)` 中，除了预测 T 块顶点配准误差 `err` 之外，加入轻量级 **推块接触引导项（Agent-to-Block Push-Side Attraction）**：当 T 块尚未对齐目标时，鼓励机械臂末端靠近 T 块背向目标一侧的推力作用点，避免机械臂远离 T 块时梯度消失；
   - 确保在 50 个测试种子上：
     - `Behavior Cloning` 取得扎实的基准分数与成功率；
     - `GPC-RANK`、`GPC-OPT`、`GPC-RANK+OPT` 均显著优于 `Behavior Cloning`；
     - **`Ours (128D Mamba-WM + Late-Stage In-Flow τ_start=0.7)`** 在最终 IoU / 成功率上超越 `Behavior Cloning`、`Early In-Flow (τ=0.0)`、`GPC-RANK` 和 `GPC-OPT`，同时比 `GPC-OPT` / `GPC-RANK+OPT` **快 4~8 倍以上**，并在 $\tau_{\text{start}} \in \{0.0, 0.3, 0.5, 0.7, 0.9, 1.0\}$ 消融表上跑出以 $\tau_{\text{start}}=0.7$ 为性能峰值的倒 U 型曲线！

---

## 3. 报告更新要求
完成修复、GPU 训练与全套评测后，更新：
1. `outputs/Phase2_GPC_Mamba_实验报告.md`
2. `outputs/Phase1_实验报告.md`
3. `outputs/evaluation/results.md`
确保所有 JSON 日志与 Markdown 报告数据完全同步、真实可复现。
