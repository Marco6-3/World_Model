# Phase 1 修复后实验报告

完成校准、120 条回合采集、正式训练和 99 个独立闭环配置。静态 M1 门槛为 3/3 成功，主矩阵在另一组种子上为 3/3。主矩阵快速档 B1 的夹碰为 0/3。

## 实现

- 策略接管区间从 pre_entry 后开始，到 release_item 完成时结束。脚本开盖、抓取、进场、退出和关盖在专家掩码之外。恢复中的撤回与重开属于受控插入。
- 物块进入盒内释放区域时，夹爪锁存张开 24 个控制步。成功判定仍要求物块落入目标区并在退出后留在盒内。
- 夹碰冲量统计盒盖与物块的接触，以及非关盖阶段盒盖与右臂的接触。单物理步阈值保持 0.004 N·s。
- 左手按压点移到后杠杆外侧，让快速档保持足够的实际开度。恢复分支缩短了回落期间的探入时间。三档弹簧阻尼参数沿用首轮。
- Flow 使用锚点插值和 1D 卷积、相邻动作差分正则与时间相关噪声。33.3 ms 控制步内，每只手的目标变化上限为 7.5 mm。因果评分和执行使用同一位置投影。

## 校准与采集

| 回弹档 | 松手后净空低于 5 mm | 直接插入夹碰 | 延迟插入夹碰 |
|---|---|---|---|
| static | 2.5 s 内未触发 | 无 | 无 |
| slow | 0.933 s | 无 | 有 |
| fast | 0.083 s | 有 | 有 |

正式数据覆盖种子 100–109，合计 98,937 个控制步。逐文件核验了专家标记和插入区间。训练、验证、离线测试分别采用 6、2、2 个组种子。

| 快速档分支 | 物理复核成功 | 物理复核夹碰 | 正式采集成功 | 正式采集夹碰 | 正式专家控制步 |
|---|---:|---:|---:|---:|---:|
| hold | 3/3 | 0/3 | 10/10 | 0/10 | 1654 |
| recover | 3/3 | 0/3 | 10/10 | 0/10 | 3674 |

[校准](repair_v2/calibration/calibration.json) · [采集清单](repair_v2/dataset/rollouts.json)

## 训练与门槛

后验加权世界模型在留出插入窗口上的净空 MAE 为 3.87 mm，夹碰精确率 85.8%，召回率 89.2%。首轮离线统计也包括插入前后的脚本窗口，两轮的评估范围有区别。

留出种子 108、109 的静态档 M4-b，在前 8 步的相邻右手目标跳变中位数由首轮 52.6 mm 降至 7.50 mm，限速前为 7.64 mm。六组探针中全块最大跳变为 7.50 mm。

正式门槛在种子 55–57 上得到静态 M1 成功 3/3，快速 B1 夹碰 0/3。门槛通过后，主矩阵使用种子 60–62。

[训练指标](repair_v2/trained/training.json) · [动作探针](repair_v2/trained/action_probe.json) · [正式门槛](repair_v2/gate/gate.json)

## 闭环与消融

默认 τ_start=0.7 的 M4-b 成功 9/9、夹碰 3/9。M1 成功 9/9、夹碰 2/9。M4-b 和 M3 平均采样耗时分别为 30.77 和 10.49 ms/块。延迟在 CPU 多工作进程闭环中采集，各档每个方法有三个回合。

快速档 M4-b 的夹碰为 0/3，M1 为 1/3。慢速档方向相反，M4-b 为 3/3，M1 为 1/3。六档消融的成功均为 9/9，夹碰数随起始时刻从 0.0 到 1.0 分别为 2、3、3、3、4、4。当前样本支持区分不同档位上的表现，统计范围为本轮三个评测种子。

M2 的 10 个实际夹碰执行前缀中，9 个的预测峰值低于碰撞阈值。

[完整矩阵与消融](evaluation/results_v2.md) · [逐回合数据](evaluation/matrix_v2.json) · [首轮报告](Phase1_实验报告_v1.md)

## 复跑

从工作区根目录执行。复跑使用新输出目录。默认读取本机已有 ALOHA USD，可用 --robot-usd 指向其他位置。闭环使用一阶梯度近似，精确雅可比入口保留。

```powershell
$python = 'C:\isaacsim_env\Scripts\python.exe'
$p = 'phase1_spring_lid'
$out = "$p/outputs/replay_v2"
& $python "$p/run.py" calibrate --out "$out/calibration"
& $python "$p/run.py" collect --out "$out/dataset" --first-seed 100 --groups 10
& $python "$p/learn.py" --raw "$out/dataset" --out "$out/trained" --train 100 101 102 103 104 105 --val 106 107 --test 108 109 --steps 1800
& $python "$p/inference.py" --raw "$out/dataset" --out "$out/trained" --model "$out/trained/model.pt" --seeds 108 109
if ($LASTEXITCODE -ne 0) { throw 'Action continuity gate failed' }
& $python "$p/run.py" gate --out "$out/gate" --model "$out/trained/model.pt" --eval-seeds 55 56 57
if ($LASTEXITCODE -ne 0) { throw 'Closed-loop gate failed' }
& $python "$p/run.py" evaluate --out "$out/evaluation" --model "$out/trained/model.pt" --eval-seeds 60 61 62
```
