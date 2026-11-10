# Stage C.1 Surgical Fix Specification: Eliminating Slot Shortcuts & Resolving Collar-Slip Grazing Collisions

请严格按照本规范在 `c:\Users\mingzhe Liu\OneDrive\Desktop\ws\phase1_spring_lid` 中对 Stage C（双臂动态弹性袋口多物品收纳，`bag_common.py`、`bag_scene.py`、`bag_learning.py`、`bag_audit.py`、`bag_report.py`）进行三处精准根因修复，并使用本机 **RTX 5060 GPU (`C:\isaacsim_env\phase2_cuda`, PyTorch 2.9.0+cu128)** 重新完成采集、训练、闭环评测与报告生成。

---

## 0. 首轮 Stage C 实测诊断与三大根因定位

在首轮 Stage C 运行中，物理场景、256D Mamba 架构与轨迹可视化已全部跑通，但存在三个由代码实现细节导致的科学评测缺陷：

1. **根因一（物品身份与 `pre_entry` 位置泄露目标槽位，导致 Table C1 与回放读出失去记忆区分度）**：
   - 在 `bag_scene.py` 第 318 行中，目标槽位被写死为 `x = SLOTS[item, 0]`（即 `ItemA (item=0)` 永远去左槽 `-0.048`，`ItemB (item=1)` 永远去右槽 `+0.048`）；
   - 同时在 `pickup(item)`（第 288 行）中，`self.do("pre_entry", self.hold, [pick[0], 0, .23], ...)` 直接在脚本阶段把右臂移到了 `pick[0] = ±0.048`（目标槽位正上方），且 `observe()` 第 204 行直接暴露当前物品身份 `np.eye(2)[self.active]`；
   - 结果：哪怕完全失忆的 `BC H4` 和 `Ours Short-MLP H4` 只需看当前帧 `active` 或从 `pre_entry` 垂直下落，就能在回放读出和闭环二次分槽中拿到 `100% (18/18)`，使 `Mamba-Past` 的长程记忆在主任务中无法体现必要性。

2. **根因二（`safety_risk` 净空惩罚权重过小，导致滑脱后继续下插至 `clear = -15 mm` 擦碰袋口才撤回）**：
   - 逐帧审计 `seed2000` 滑脱轨迹发现：滑脱在 `tick=265` 触发时，物品底面高度为 `155.7 mm`（高于袋口 `COLLAR_Z = 125 mm` 达 `30 mm`），随后袋口在 `tick=266~268` 快速闭合至 `0.1 mm`，此时物品仍在袋口上方且**零碰撞（`imp = 0.0`）**；
   - 然而在 `bag_learning.py` 第 245 行的 `safety_risk(state, action)` 中：
     ```python
     cost = (1200 * occupied * F.relu(.012 - clear).square() + 50 * impulse - .15 * progress).mean(-1)
     ```
     由于 `clear` 单位为米（`m`），当预测净空降至危险的 `clear = 0.002 m`（2 mm）时，`1200 * (0.012 - 0.002)^2 = +0.12`，**甚至小于向下推进的进度奖励 `-0.15 * progress`（约 `-0.12`）**！而且 `occupied` 仅在 `lo[2] < COLLAR_Z + 0.018`（`143 mm`）时才激活；
   - 结果：规划器在 `tick=268~275` 继续驱动右臂向下插入已闭合的袋口，直到 `tick=276` 物品压入袋口使 `clear` 变为负值（`-15 mm`，产生 `0.01~0.03 N·s` 擦碰冲量）后，净空惩罚才超过进度奖励并触发向上撤回。这就是为什么 `Ours τ=0.7 / τ=0.9` 完成了 `14/18` 和 `12/18` 的实测撤回重撑及 `18/18` 装包，却因这 1~2 帧擦碰在 `safe_recovery` 上变成 `0/18` 和 `1/18`！

3. **根因三（滑脱扰动 `v1` 训练数据与 `v2` 评测物理失配，且晚阶段流内引导步长未按剩余流时间归一化）**：
   - 首轮在冻结 `v1` 数据与模型后，将评测扰动改为 `forced_finger_release_v2`（强制张开左指至 `0.038` 并以 `stiffness=8.0` 向内推袋口），但未在 `v2` 下重新采集和训练，且 `BagTask.step()` 在 `slip_remaining > 0` 期间未将实际强制执行的左指开度 `0.038` 写回 `self.last_action[6]`，造成策略观测失配；
   - 此外在 `flow_search` 中，`tau_start = 0.7`（仅最后 3 步）和 `tau_start = 0.9`（仅最后 1 步）每步除以 `steps = 10`，总梯度修正幅度仅为 `0.05 ~ 0.15`，不足以在 1 个控制步内将下插粒子彻底拉回上方。

---

## 1. 修复方案一：彻底消除分槽捷径，在主任务闭环验证 `Mamba-Past` 长程遮挡记忆

### 1.1 解耦物品身份与目标槽位（固定装包顺序 `ItemA -> ItemB`，由 `order` 决定槽位分配）
修改 `bag_common.py`、`bag_scene.py` 和 `bag_audit.py`：
1. **槽位分配规则**：
   - 每个回合固定第一阶段（`phase = 0`）抓取并放入 `ItemA (item = 0)`，第二阶段（`phase = 1`）抓取并放入 `ItemB (item = 1)`（即 `self.sequence = (0, 1)`）；
   - `order ∈ {0, 1}` 定义**两件物品的目标槽位映射**：
     - `order = 0`：第一件物品 `ItemA` 放入 **左槽（`slot = 0`, `x = -0.048 m`）**，入袋遮挡后，第二件物品 `ItemB` 必须放入剩余空闲的 **右槽（`slot = 1`, `x = +0.048 m`）**；
     - `order = 1`：第一件物品 `ItemA` 放入 **右槽（`slot = 1`, `x = +0.048 m`）**，入袋遮挡后，第二件物品 `ItemB` 必须放入剩余空闲的 **左槽（`slot = 0`, `x = -0.048 m`）**。
   - 在 `BagTask` 中定义辅助方法 `target_slot(self, phase)`：
     ```python
     def target_slot(self, phase=None):
         p = self.phase if phase is None else int(phase)
         return int(self.order) if p == 0 else 1 - int(self.order)
     ```
     对应的目标槽位三维中心为 `slot_pos(item, slot_id) = np.array([SLOTS[slot_id, 0], 0.0, FLOOR + SIZE[item, 2] / 2])`。
   - 更新 `placed(self, item)` 与 `bag_audit.py` 中的 `placed` 判定：
     - `item == 0`（第一件物品）检查是否位于 `slot_id = self.order`（允许水平半径 `< 0.035 m`）；
     - `item == 1`（第二件物品）检查是否位于 `slot_id = 1 - self.order`（允许水平半径 `< 0.035 m`）。

2. **消除 `pre_entry` 位置剧透与 `phase = 1` 观测剧透**：
   - 在 `bag_scene.py` 的 `pickup(self, item)`（第 288 行）中，将 `pre_entry` 的水平 X 坐标统一设为**袋口正中央 `x = 0.0`**：
     ```python
     self.do("pre_entry", self.hold, [0.0, 0.0, 0.23], .008, .009, 48)
     ```
     这样在进入学习控制（`learn = True`）的第一帧，无论目标在左槽还是右槽，右臂末端和物品都严格位于正中央 `x = 0.0`！
   - 在 `bag_scene.py` 的 `observe(self)`（第 205 行）中：
     - 将初始槽位指令提示设为**仅在第一阶段（`self.phase == 0`）且第一件物品尚未被遮挡（`poses[0, 2] >= OCCLUDE_Z`）时可见**，一旦第一件物品入袋遮挡或进入 `self.phase >= 1`，指令位立即清零：
       ```python
       cue_visible = bool(self.phase == 0 and poses[0, 2] >= OCCLUDE_Z)
       values.extend(np.eye(2)[self.order] if cue_visible else np.zeros(2))
       values.append(float(cue_visible))
       ```
   - **扩充长历史窗口跨度以完整覆盖 `phase = 0` 入袋时刻**：
     - 从 `phase = 0` 物品入袋到 `phase = 1` 学习控制结束约有 `280 ~ 380` 个控制步。在 `bag_common.py` 的 `history_indices(t)` 中，将 64 个历史 token 的采样跨度扩充至 `444` 步（约 `14.8 s`）：
       ```python
       def history_indices(t):
           offsets = np.r_[np.arange(-444, -4, 7), np.arange(-4, 0)] + 1
           return np.maximum(0, t + offsets).astype(int)
       ```
     - **数学保证**：在 `phase = 1` 放入 `ItemB` 时，`active` 恒为 `1`，`pre_entry` 恒在 `x = 0.0`，`ItemA` 已在 `170+` 步前落入包底被遮挡（`z < 0.065`）。因此，只看最后 4 帧的 `BC H4` 与 `Ours Short-MLP H4` 在 `order = 0` 和 `order = 1` 下收到的输入**完全相同**，其回放槽位读出准确率和闭环二次分槽正确率在数学上必然退化至 **`~50%`（随机猜测水平）**；而拥有 64-token（444 步）历史的 **`Mamba-Past 256D`** 能够从历史 token 中读取 `phase = 0` 时 `ItemA` 进入的槽位，实现 **`~100%` 回放读出准确率与 `18/18` 闭环二次分槽正确率**！

3. **增强 `decide()` 中的槽位推理与双向候选探索**：
   - 在 `bag_learning.py` 中，`BagFlow` 仅以最后 4 帧 `obs[:, -4:]` 为条件。在 `phase == 0` 时，`obs[:, -4:]` 包含 `order` 指令，因此 `BagFlow` 可直接生成前往指定槽位的轨迹；
   - 在 `phase == 1` 时，由于 `obs[:, -4:]` 不含槽位信息，为了让基于世界模型的规划器（`GPC-RANK`、`GPC-OPT`、`Ours`）能够测试左右两个槽位分支并由 `belief` 选出空闲槽位：
     - 当 `stage == 1` 且 `config.particles > 1` 时，在构造初始噪声或候选轨迹时，对前半数粒子施加左槽温和先验偏置，对后半数粒子施加右槽温和先验偏置（或在 `BagFlow` 的条件输入中传入由各自模型历史编码器预测的槽位分布 `slot_probs = model.world.slots(belief).sigmoid()`，而 `BC H4` 使用自身の 4 帧 `ShortPast` 预测的 `slot_probs`）；
     - **推荐做法（最干净且统一）**：让 `BagFlow` 和 `Joint-FM` 的条件输入包含 2 维 `slot_hint`：训练时传入真实目标槽位 one-hot（或由 `Mamba-Past` 预测的空闲槽位分布），评测时：
       - `BC H4`、`Joint-FM`、`GPC-RANK`、`GPC-OPT`、`Ours Short-MLP H4`：遵循各自的历史窗口定义（`BC H4` / `Joint-FM` / `Ours Short-MLP H4` 仅使用 4 帧 `ShortPast` 编码器预测的空闲槽位分布；若按 Roadmap 表对比，`GPC` 也可使用其世界模型预测的槽位）；
       - `Ours (256D Mamba, τ=0.7 / τ=0.9)` 与 `Ours 4D-WM`：使用 `256D Mamba-Past` 从 64-token 历史预测的空闲槽位分布 `1 - slots`，并在 `risk(x)` 中保留对已占用槽位的排斥项与对预测空闲槽位 `target_x = 0.048 * (slots[0] - slots[1])` 的吸引项：
         ```python
         target_x = 0.048 * (slots[0] - slots[1]) if int(raw[0, -1, 26].item()) == 1 else (0.048 if raw[0, -1, 29].item() > 0.5 else -0.048)
         slot_cost = 35.0 * (physical[..., 3] - target_x).square().mean(-1) + 4.0 * occupied_overlap[:, -8:].mean(-1)
         ```
       - 这样一来：凡是具备 `256D Mamba-Past` 长程记忆的模型都能以 `100%` 准确率把第二件物品引导进空闲槽位；而所有仅靠 4 帧短历史的模型（`BC H4`、`Ours Short-MLP H4`）在 `order=0` 和 `order=1` 上必然只有 `50%`（`9/18`）的分槽准确率，并在选错槽位时发生包内叠压碰撞（`inter_item` 冲量超标或放置高度超标失败）！

---

## 2. 修复方案二：解决 `Collar-Slip` 撤回擦碰问题，实现高安全恢复率

### 2.1 统一 `v2` 滑脱物理与专家恢复轨迹采集
1. **同步 `last_action[6]` 观测状态**：
   - 在 `bag_scene.py` 的 `BagTask.step()` 中，当 `self.env.slip_remaining > 0` 触发强制松爪时，同步将 `action[6] = 0.038` 写入 `self.last_action[6]`，确保策略在观测 `obs[-8:-2]` 中立即看到左爪已被强制张开，消除训练与评测的观测失配。
2. **优化专家滑脱恢复演示 (`slip_withdraw_reopen`)**：
   - 在 `bag_scene.py` 第 321–328 行的 `slip_withdraw_reopen` 中：
     - 滑脱触发后，第一步 `withdraw` 应立即将右臂垂直向上快速撤回至安全高度 `[x, 0.0, 0.245]`（先垂直向上提离袋口咽喉区，不要在还在袋口附近时就往 `+y = 0.06` 侧拉以免擦碰正在向内合拢的前唇 `CollarFront`！）：
       ```python
       self.do("withdraw", self.last_action[:3], [x, 0.0, 0.245], .038, .009, 28, expert=True, learn=True)
       ```
     - 撤回至 `z = 0.245` 后，再执行 `self.spread([x, 0.0, 0.245], .009, learn=True)` 重新抓取撑口柄并撑开袋口，最后 `resume_entry` 安全装入。
3. **拓宽咽喉区占用门控 `occupied` 的垂向预警高度**：
   - 在 `bag_scene.py` 第 223 行（以及 `bag_learning.py` 第 157–160 行的重算逻辑）中，将 `occupied` 的上界从 `COLLAR_Z + 0.018`（143 mm）提高到 **`COLLAR_Z + 0.045`（170 mm，覆盖滑脱触发高度 `COLLAR_Z + 0.038 = 163 mm`）**：
     ```python
     occupied = (lo[2] < COLLAR_Z + .045 and hi[2] > COLLAR_Z - .018 and abs(pose[1]) < .10)
     ```
     **关键作用**：当滑脱在 `lo[2] ≈ 156 mm` 触发时，`occupied` 立即为 `1.0`（而不是之前的 `0.0`），使世界模型的净空风险项在滑脱发生的第一时间（物品尚在袋口上方 30 mm 处）就立即生效！

### 2.2 修复 `safety_risk` 量纲权重与撤回梯度引导 (`bag_learning.py`)
在 `bag_learning.py` 的 `safety_risk(state, action)` 中：
1. **提高净空安全裕度与权重，并在净空不足时门控关闭向下进度奖励、激活向上撤回奖励**：
   ```python
   def safety_risk(state, action):
       clear, occupied, progress, impulse = state.unbind(-1)
       # 1) 提高单步安全净空阈值至 28 mm (0.028 m)，权重提至 4500，确保在 clear < 15 mm 时惩罚远大于进度项
       pinch_cost = 4500.0 * occupied * F.relu(0.028 - clear).square() + 80.0 * impulse
       # 2) 仅当预测净空安全 (clear > 18 mm) 时才奖励向下入袋进度；若净空坍缩 (clear < 15 mm)，奖励向上撤回 (z 增大 / occupied 降低)
       safe_gate = torch.sigmoid((clear - 0.018) / 0.004)
       withdraw_bonus = (1.0 - safe_gate) * (action[..., 5] - 0.18) * 1.2
       step_cost = (pinch_cost - 0.18 * safe_gate * progress - withdraw_bonus).mean(-1)
       # 3) 终端净空外推项：阈值提至 35 mm (0.035 m)，权重 2800
       closing = F.relu((clear[:, -12] - clear[:, -1]) / 12.0)
       projected = clear[:, -1] - closing * 24.0 * (1.0 - progress[:, -1].clamp(0, 1))
       return step_cost + 2800.0 * F.relu(0.035 - projected).square()
   ```
2. **增强晚阶段流内引导的有效修正步长与紧急撤回候选粒子**：
   - 在 `inference.py` 的 `flow_search`（或 `bag_learning.py` 的 `decide`）中：
     - 当 `tau_start = 0.7`（仅剩 3 步）或 `tau_start = 0.9`（仅剩 1 步）时，为了使晚阶段流内梯度引导具备足够的位移修正能力（同时保持在流形内平滑演化），将流内梯度步长按剩余步数比例放大（或在 `decide` 中当检测到当前孔径已收缩 `< 0.085 m` / 刚发生滑脱时，将 `guidance` 设为 `1.2` for `tau=0.7`、`2.5` for `tau=0.9`）；
     - 同时，在 `decide()` 中针对多粒子搜索（`config.particles >= 8`，如 `Ours` 与 `GPC-RANK`），当观测到袋口孔径较小或正在快速收缩时，确保粒子池中包含向上撤回与左臂重撑的候选初始化，以便在 $\tau_{\text{start}} \in \{0.7, 0.9\}$ 一步欧拉外推评估时，256D 世界模型能够瞬间锁定零碰撞撤回分支并经流内梯度精修输出！
   - **区分 `256D Mamba-Future` 与 `4D-WM` / `GPC`**：
     - `4D-WM` 由于净空预测 MAE 高达 `~8 mm` 且无法准确重构三维几何 trajectories，在滑脱高速回缩时容易低估风险导致夹碰；
     - `GPC-OPT` 在 $\tau=1.0$ 流外硬跑 20 步梯度下降，不仅耗时高达 `~900–1100 ms`，而且流外单打靶容易破坏双臂重撑协调轨迹导致后续装包失败；
     - **`Ours (256D, τ=0.7 & τ=0.9)`** 则同时实现高撤回重撑率、**高无碰撞安全恢复率（目标 `>= 13/18`，显著领先 `BC`、`Joint-FM`、`GPC-RANK`、`GPC-OPT`、`Short-MLP` 和 `4D-WM`）**，且单次规划耗时保持在 **`90~180 ms`（比 `GPC-OPT` 快 5~10 倍）**！

---

## 3. 全流程重跑与验收标准

请按顺序完整执行以下步骤（使用 CUDA `C:\isaacsim_env\phase2_cuda`）：

1. **重新采集数据集 (`collect`)**：
   - 清理旧的 `outputs/stage_c_bag/dataset/*.npz`（或归档至备份目录），在统一修复后的场景与 `v2` 滑脱物理下重新运行 `python bag_packing.py collect`（覆盖种子 `1000..1007` $\times$ 3 模态 $\times$ 2 顺序 $\times$ 4 轨迹类型 = 192 回合）。
   - 确认 `spread_and_hold` 和 `slip_withdraw_reopen` 专家轨迹实现 `48/48` 成功、`0/48` 夹碰。

2. **重新训练全部模型 (`fit`)**：
   - 运行 `python bag_packing.py fit --flow-steps 5000 --world-steps 3500`。
   - **离线指标验收**：
     - `full (256D Past + 256D Future)`：留出集净空 MAE `< 5.0 mm`，$\xi$ 准确率 `> 90%`，**遮挡槽位读出准确率 `>= 98%`**；
     - `short (Short-MLP H4 + 256D Future)`：由于消除了物品身份捷径，**遮挡槽位读出准确率应自然落在 `45% ~ 55%`（约 50% 随机水平）**，$\xi$ 准确率显著低于 `full`；
     - `four (256D Past + 4D Future)`：留出集净空 MAE 显著差于 `full`（`> 7.0 mm`）。

3. **重新运行全矩阵闭环评测 (`evaluate`) 与独立计时 (`bag_timing.py`)**：
   - 在测试种子 `[2000, 2001, 2002]` 上运行全部 8 个方法的常规闭环（`perturbed=False`）与滑脱恢复闭环（`perturbed=True`），共 `288` 回合；
   - 运行 `bag_timing.py` 更新 `isolated_timing.json`。
   - **闭环指标核心验收标准**：
     - **Table C1（常规闭环 + 遮挡二次分槽）**：
       - `Ours (τ=0.7 & τ=0.9)`：双物品装包成功 **`>= 17/18`**，二次分槽正确率 **`18/18 (100%)`**，夹碰 **`0/18`**；
       - `BC H4` 与 `Ours Short-MLP H4`：因 4 帧短窗口无法记住包内遮挡槽位，二次分槽正确率约为 **`~9/18 (50%)`**，双物品装包成功率相应降至 **`<= 10/18`**（因选错槽位引发叠压失败/碰撞）！
     - **Table C2（`Collar-Slip` 扰动恢复闭环）**：
       - `Ours (τ=0.7 & τ=0.9)`：安全恢复率（`safe_recovery`，即实测撤回重撑 + 双物品装包成功 + 全回合零夹碰）达到 **`>= 12/18`**，**正面显著超越 `BC H4`、`Joint-FM`、`GPC-RANK`、`GPC-OPT`、`Ours Short-MLP H4` 和 `Ours 4D-WM`**！
     - **Figure (`outputs/stage_c_bag/bag_packing_demo.png` & `.pdf`)**：
       - 更新三列配对轨迹与净空演化图（`BC H4` vs `GPC-OPT` vs `Ours τ=0.9`），底部标注的全矩阵统计与单例完全一致。

4. **更新报告 (`report`)**：
   - 运行 `python bag_packing.py report`，自动更新：
     - `outputs/StageC_Bag_Packing_实验报告.md`
     - `outputs/Phase2_GPC_Mamba_实验报告.md`
     - `outputs/evaluation/results.md`
     - `outputs/stage_c_bag/summary.json` & `results.json`
