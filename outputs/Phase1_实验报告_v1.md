# Phase 1 首轮实验报告

已完成三档盒盖标定、混合交互采集、监督训练和 33 个独立闭环配置。默认混合引导 M4-b 在三档回弹中均未完成任务，并均触发夹碰阈值。本轮结果未显示因果引导提高装盒成功率或降低夹碰率。

训练、验证、离线测试分别使用种子 10、11、12。闭环使用独立种子 40，每种方法每档一回合。这是一轮端到端试验，样本量不足以判断统计显著性。

## Step 1　物理标定

IsaacLab / PhysX CPU 场景加载 ALOHA 双臂和具有实体碰撞几何的盒盖铰链。重力平衡后的弹簧阻尼扭矩驱动盒盖回落。下表为隔离的刚体夹具结果，三档均从 0.83 rad 松手。

| 档位 | 刚度 N·m/rad | 阻尼 N·m·s/rad | 净空低于 5 mm | 直接夹碰 | 延迟夹碰 | 撤回夹碰 |
|---|---:|---:|---|---|---|---|
| static | 0.000 | 0.012 | 2.5 s 内未触发 | 无 | 无 | 无 |
| slow | 0.060 | 0.035 | 0.933 s | 无 | 有 | 无 |
| fast | 0.620 | 0.022 | 0.083 s | 有 | 有 | 无 |

[物理标定和逐步数据索引](calibration/calibration.json)

## Step 2　混合交互数据

36 条回合共 29,376 个控制步。动作是双手末端位置和右夹爪开度，底层 IK 转为关节控制。物体通过实际夹爪接触抓取。四维标签来自刚体位姿和接触冲量。控制间隔 1/30 s，历史 12 步，预测 24 步。相同起点的分支按种子分在同一集合。分支通过相同种子和动作前缀重放，PhysX 重放存在小幅数值差异。

| 档位 | 动作分支 | 成功 | 夹碰 | 最大冲量 N·s |
|---|---|---:|---:|---:|
| static | direct | 3/3 | 0/3 | 0.000 |
| static | hold | 3/3 | 0/3 | 0.000 |
| static | delayed | 3/3 | 0/3 | 0.000 |
| static | recover | 3/3 | 0/3 | 0.000 |
| slow | direct | 3/3 | 3/3 | 0.090 |
| slow | hold | 3/3 | 0/3 | 0.000 |
| slow | delayed | 0/3 | 3/3 | 1.079 |
| slow | recover | 3/3 | 0/3 | 0.000 |
| fast | direct | 3/3 | 3/3 | 0.346 |
| fast | hold | 3/3 | 3/3 | 0.026 |
| fast | delayed | 0/3 | 3/3 | 0.463 |
| fast | recover | 3/3 | 3/3 | 0.634 |

夹碰按任一物理步的非预期盒盖接触冲量达到 0.004 N·s 计数。快速档可以带着接触完成装盒，因此任务成功和夹碰分别统计。恢复分支撤出后通道占用归零，快速档重新插入或后续仍可能夹碰。

[采集清单](dataset/rollouts.json)

## Step 3　训练与离线检查

训练、验证、测试分别包含 1753、1748、1755 个重叠窗口。FM 与联合生成基线使用同一批专家动作窗口，世界模型学习全部有效分支，GRU 按动作前缀预测。

| 数据条件 | 净空 MAE mm | 占用时净空 MAE mm | 夹碰精确率 | 夹碰召回率 |
|---|---:|---:|---:|---:|
| 验证，已知回弹档 | 5.17 | 8.33 | 88.7% | 93.2% |
| 验证，后验加权 | 6.16 | 8.39 | 88.2% | 94.2% |
| 测试，已知回弹档 | 5.18 | 8.59 | 90.4% | 92.7% |
| 测试，后验加权 | 6.24 | 8.82 | 89.6% | 93.3% |

测试组回弹档位分类准确率为 61.3%。配对动作分支的净空差值 MAE 为 14.46 mm，有明确冲量差异的 2 对中正确排序 2 对。

[训练日志指标](trained/training.json) · [后验加权指标](trained/posterior_validation.json)

## Step 4　闭环矩阵与消融

开盖、抓取、放置后的退出及关盖共用脚本。学习方法控制插入和恢复，策略接收带可见性掩码的几何观测。冻结采样步数为 10，默认 τ_start = 0.7，闭环使用一阶梯度近似。精确雅可比路径用训练检查点单独验证。全部方法和消融使用同一评估种子及回弹档。

M4-b 平均 15.09 ms/块和 4 批世界模型调用，M3 平均 4.20 ms/块和 1 批调用。六档消融均无成功回合。早停失败会拉低平均回合时间和代价。

M2 的 33 个实际夹碰执行前缀中，0 个的联合预测峰值低于碰撞阈值。本轮未复现阈值意义上的虚假安全预测。

验证组静态档使用相同入盒前观测采样 16 次，M4-b 前 8 步右手相邻目标跳变的中位数为 52.6 mm。该现象暴露了动作连续性缺口。

[完整闭环表](evaluation/results_v1.md) · [逐回合数据](evaluation/matrix_v1.json) · [动作连续性检查](trained/action_probe.json)

## 复跑

在工作区根目录运行以下命令。默认加载本机已有的 ALOHA USD，其他位置可传入 --robot-usd。使用新目录保留各轮结果。Isaac 日志中有 ALOHA visual prim 引用警告，本轮核验覆盖 CPU 物理和数值输出。

```powershell
$python = 'C:\isaacsim_env\Scripts\python.exe'
& $python phase1_spring_lid/run.py calibrate --out phase1_spring_lid/outputs/replay_calibration
& $python phase1_spring_lid/run.py collect --out phase1_spring_lid/outputs/replay_dataset --first-seed 10 --groups 3
& $python phase1_spring_lid/learn.py --raw phase1_spring_lid/outputs/replay_dataset --out phase1_spring_lid/outputs/replay_trained --train 10 --val 11 --test 12 --steps 1250
& $python phase1_spring_lid/run.py evaluate --out phase1_spring_lid/outputs/replay_evaluation --model phase1_spring_lid/outputs/replay_trained/model.pt --eval-seeds 40
```
