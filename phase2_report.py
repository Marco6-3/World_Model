"""Render the three Chinese reports from completed Phase 2 evaluation JSON."""

import argparse
import json
from pathlib import Path

import numpy as np


TAUS = (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)
REGIMES = ("static", "slow", "fast")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def merge_shards(push_dir, spring_dir):
    push = [read(path) for path in sorted(push_dir.glob("shard_*/results.json"))]
    spring = [read(path) for path in sorted(spring_dir.glob("seed*/closed_loop.json"))]
    if len(push) != 4 or len(spring) != 3:
        raise ValueError("Expected four completed Push-T shards and three completed packing shards")
    cfg = push[0]["config"]
    if any(row["config"] != cfg or row["methods"] != push[0]["methods"] for row in push):
        raise ValueError("Push-T shard settings differ")
    seeds = sorted(seed for row in push for seed in row["seeds"])
    if len(seeds) != len(set(seeds)):
        raise ValueError("Repeated Push-T seed across shards")
    data = dict(config=cfg, seeds=seeds, methods=push[0]["methods"],
                episodes=[episode for row in push for episode in row["episodes"]],
                shard_sizes=[len(row["seeds"]) for row in push])
    base = {k: v for k, v in spring[0].items() if k not in ("episodes", "seeds")}
    if any({k: v for k, v in row.items() if k not in ("episodes", "seeds")} != base for row in spring):
        raise ValueError("Packing shard settings differ")
    spring_seeds = sorted(seed for row in spring for seed in row["seeds"])
    if len(spring_seeds) != len(set(spring_seeds)):
        raise ValueError("Repeated packing seed across shards")
    episodes = [episode for row in spring for episode in row["episodes"]]
    for dim in base["dimensions"]:
        if len({row["weights_sha256"] for row in episodes if row["d_z"] == dim}) != 1:
            raise ValueError(f"Packing checkpoint mismatch at Dz={dim}")
    closed = dict(**base, seeds=spring_seeds, episodes=episodes, shard_sizes=[len(row["seeds"]) for row in spring])
    for path, result in ((push_dir / "results.json", data), (spring_dir / "closed_loop.json", closed)):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        tmp.replace(path)


def table(headers, rows):
    return "\n".join("| " + " | ".join(map(str, row)) + " |"
                     for row in (headers, ["---"] * len(headers), *rows))


def group(rows, key, value):
    selected = [row for row in rows if row[key] == value]
    if not selected:
        raise ValueError(f"Empty required cell for {key}={value}")
    return selected


def average(rows, key):
    return float(np.mean([row[key] for row in rows]))


def count(rows, key):
    return f"{sum(bool(row[key]) for row in rows)}/{len(rows)}"


def packed_samples(rows, key):
    return np.array([sample[key] for row in rows for sample in row["chunks"]])


def push_latency(rows):
    return np.array([v for row in rows for v in row["decision_ms"]])


def push_row(name, rows):
    conf = rows[0]["config"]
    ms = push_latency(rows)
    jumps = [np.mean(row["jump_px"]) for row in rows if row["jump_px"]]
    return [name, conf["particles"], conf["opt_steps"], f'{average(rows, "final_iou"):.3f}',
            f'{average(rows, "max_iou"):.3f}', f'{average(rows, "coverage"):.3f}', count(rows, "success"),
            f'{np.mean(jumps):.2f}' if jumps else "—",
            f'{np.mean([v for row in rows for v in row["wm_calls"]]):.1f}',
            f'{np.mean([v for row in rows for v in row["wm_rollouts"]]):.1f}',
            f"{ms.mean():.1f} / {np.median(ms):.1f} / {np.quantile(ms, 0.95):.1f}"]


def paired_ci(ours, other):
    baseline = {row["seed"]: row["final_iou"] for row in other}
    delta = np.array([row["final_iou"] - baseline[row["seed"]] for row in ours])
    indices = np.random.default_rng(450).integers(0, len(delta), (10000, len(delta)))
    lo, hi = np.quantile(delta[indices].mean(1), [0.025, 0.975])
    return f"{delta.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"


def push_report(data, train):
    rows = data["episodes"]
    methods = ["Behavior Cloning", "GPC-RANK K=16", "GPC-RANK K=64", "GPC-OPT",
               "GPC-RANK+OPT", "Early In-Flow", "Ours"]
    by_method = {name: group(rows, "method", name) for name in data["methods"]}
    seeds = data["seeds"]
    if len(set(seeds)) < 50 or any(sorted(row["seed"] for row in arm) != sorted(seeds) for arm in by_method.values()):
        raise ValueError("Push-T requires all methods to finish the same 50 or more seeds")
    cfg = data["config"]
    arm = by_method["Ours"]
    best = max(methods, key=lambda name: average(by_method[name], "final_iou"))
    source = train["expert_source"]
    official = source.startswith("https://")
    source_link = f" · [Columbia 原始回放]({source})" if official else ""
    source_label = "Columbia 原始回放" if official else source
    excitation = train.get("exploration_archive", {})
    motion = excitation.get("train_object_motion_steps")
    motion_text = "" if motion is None else f"其中有 {motion} 步的 T 块平移至少 1 px 或转动至少 0.01 rad。"
    lines = [
        "## Stage A · 标准 Push-T 状态基准",
        "",
        f"使用官方 `gym-pusht {train['gym_pusht']}` 与 Pymunk {train['pymunk']}。"
        f"专家源为 {source_label}，共 {train['expert_episodes']} 条。"
        f"按回合划分 {len(train['train_expert_ids'])} 条训练和 {len(train['val_expert_ids'])} 条验证，"
        f"另用控制器采集 {len(train['train_exploration_seeds'])} 条训练扰动和 "
        f"{len(train['val_exploration_seeds'])} 条验证扰动。训练扰动中有 "
        f"{train['training_exploration_contact_steps']}/{64 * len(train['train_exploration_seeds'])} 步含接触。{motion_text}"
        "Flow 只用专家样本，世界模型合用专家与扰动样本。扰动策略从随机初态或训练专家中途状态启动，"
        "周期性随机推向 T 块附近或扰动专家目标。本轮控制器和数据量按本地实验定义。"
        " [公开环境](https://github.com/huggingface/gym-pusht)" + source_link,
        "",
        f"各方法共享同一冻结的 256 通道条件速度场 PushTFlow 和 128 维 Mamba。Flow 训练 {train['flow_updates']} 次更新，"
        f"世界模型训练 {train['world_updates']} 次更新。世界模型根据 {cfg['history']} 帧状态预测 "
        f"{cfg['horizon']} 帧，含 T 块位姿、有限差分速度、末端位置与速度以及目标配准误差。"
        f"按相同种子 {min(seeds)}–{max(seeds)} 重置，滚动执行 {cfg['commit']} 步，"
        f"每回合最多 {cfg['max_steps']} 步。重置按角度后位置赋值；规划目标结合预测 T 块顶点配准误差与推力接触点吸引项。"
        f"在 {train['heldout_windows']} 个验证窗口上，T 块 x/y 坐标 MAE "
        f"分别为 {256 * train['heldout_state_mae'][2]:.2f} / {256 * train['heldout_state_mae'][3]:.2f} px。",
        "",
        "下表比较本地实现的 RANK、OPT 与 RANK+OPT 规划机制，并按本项目指定预算测量；不等同于原论文配置或成绩的复现。",
        "",
        table(["方法", "K", "M", "最终 IoU", "峰值 IoU", "最终覆盖率", "成功",
               "相邻动作跳变 px", "WM 批调用/决策", "WM 轨迹/决策", "延迟均值 / P50 / P95 ms"],
              [push_row(name, by_method[name]) for name in methods]),
        "",
        "覆盖率为 T 块和目标交集除以目标面积，IoU 使用并集。成功取官方超过 95% 覆盖率。"
        "分数与动作跳变先按回合计算再按种子等权汇总，只有至少两步的回合参与动作跳变统计。"
        "延迟和世界模型调用按决策汇总。"
        "延迟含观测编码、采样、规划、投影和动作返回，物理推进在计时区间之外。"
        f"评测使用 {cfg['device']}、PyTorch {cfg['torch']} 和 {cfg['torch_threads']} 个 PyTorch 线程。"
        + ("同机按种子分片并行执行，延迟反映该机器运行本批实验时的墙钟耗时。" if "shard_sizes" in data else ""),
        "",
        f"最终 IoU 最高为 {best}（{average(by_method[best], 'final_iou'):.3f}），"
        f"Ours 为 {average(arm, 'final_iou'):.3f}，成功 {count(arm, 'success')}。"
        "对比量化的是本次冻结策略与状态模型下的规划差异，性能依赖专家策略产生的候选质量。"
        "以下差值按同种子配对，括号为 10,000 次种子重采样的 95% 百分位区间。",
        "",
        table(["比较", "最终 IoU 差值（Ours 减去对照）"],
              [[name, paired_ci(arm, by_method[name])] for name in methods if name != "Ours"]),
        "",
    ]
    return "\n".join(lines), by_method


def spring_report(memory, dimensions, loop, planner_rank):
    rows = loop["episodes"]
    seeds = loop["seeds"]
    regular = [row for row in rows if not row["perturbed"]]
    normal = [row for row in regular if row["d_z"] == 128]
    recovery = [row for row in rows if row["perturbed"]]
    core = ("BC", "Joint-FM", "GPC-RANK", "GPC-OPT", "GPC-RANK+OPT", "Ours", "Ours fixed COMMIT=8")
    mem_rows, dim_rows, main_rows, recovery_rows, ranking = [], [], [], [], []
    for method, repeats in memory.items():
        held = [row["test"] for row in repeats]
        vals = [row["correct"] / row["samples"] for row in held]
        mem_rows.append([method, repeats[0]["parameters"], held[0]["samples"],
                         f"{np.mean(vals):.1%} ± {np.std(vals):.1%}",
                         f'{average(held, "angle_mae_rad"):.3f}', f'{average(held, "speed_mae_rad_s"):.3f}'])
    expected = {(regime, seed) for regime in REGIMES for seed in seeds}
    planner_rows = []
    for method in ("BC", "Joint-FM", "GPC-RANK", "GPC-OPT", "GPC-RANK+OPT", "Ours"):
        arms = group(planner_rank, "method", method)
        scores = [count(arms, "match")]
        scores.extend(count(group(arms, "regime", regime), "match") for regime in REGIMES)
        planner_rows.append([method, *scores])
    for dim in (4, 32, 128, 256):
        d = dimensions[str(dim)]
        test = d["test"]
        closed = [row for row in regular if row["d_z"] == dim and row["variant"] == "Ours"]
        if {(row["regime"], row["seed"]) for row in closed} != expected:
            raise ValueError(f"Incomplete closed-loop dimensions at Dz={dim}")
        dim_rows.append([dim, d["parameters"], f'{1e3 * test["clearance_mae_m"]:.2f}',
                         f'{test["lid_angle_mae_rad"]:.3f}', f'{test["lid_speed_mae_rad_s"]:.3f}',
                         f'{1e3 * test["visible_object_xyz_mae_m"]:.2f}',
                         f'{test["missed_steps"]}/{test["collision_steps"]} ({test["missed_steps"]/test["collision_steps"]:.1%})',
                         f'{test["false_alarm_steps"]}/{test["noncollision_steps"]}',
                         count(closed, "success"), count(closed, "pinch_or_collision")])
        offline = d["offline_test"]
        ranking.append([dim, f'{sum(r["safe_ranking_pass"] for r in offline)}/{len(offline)}',
                        ", ".join(f'{r["seed"]}/{r["regime"]} {r["predicted_choice"]}/{r["expected_safe_choice"]}'
                                  for r in offline)])
    for label in core:
        selected = group(normal, "variant", label)
        if {(row["regime"], row["seed"]) for row in selected} != expected:
            raise ValueError(f"Incomplete regular packing arm {label}")
        aggregate = [label]
        for regime in REGIMES:
            arm = group(selected, "regime", regime)
            aggregate += [count(arm, "success"), count(arm, "pinch_or_collision")]
        commit = [chunk["commit_limit"] for row in selected for chunk in row["chunks"] if "commit_limit" in chunk]
        aggregate += [f'{sum(x == 4 for x in commit)}/{len(commit)}' if commit else "0（固定 8）",
                      f'{np.median(packed_samples(selected, "ms")):.1f}']
        main_rows.append(aggregate)
        for regime in ("slow", "fast"):
            r = group(group(recovery, "variant", label), "regime", regime)
            if sorted(row["seed"] for row in r) != sorted(seeds):
                raise ValueError(f"Incomplete perturbed packing arm {label}/{regime}")
            recovery_rows.append([label, regime, count(r, "success"), count(r, "withdrew_and_reopened"),
                                  count(r, "safe_recovery"), count(r, "pinch_or_collision"),
                                  sum(row["preplan_peak_impulse_Ns"] >= 0.004 for row in r)])
    ours = group(normal, "variant", "Ours")
    fixed = group(normal, "variant", "Ours fixed COMMIT=8")
    ours_recovery = group(recovery, "variant", "Ours")
    fixed_recovery = group(recovery, "variant", "Ours fixed COMMIT=8")
    bc_recovery = group(recovery, "variant", "BC")
    threshold = dimensions["128"]["warning_clearance"]
    test = dimensions["128"]["test"]
    prefix_gap = max(r["max_prefix_feature_difference"] for r in dimensions["128"]["offline_test"])
    split = dimensions["128"]["split_seeds"]
    lines = [
        "## Stage B · 弹簧盖",
        "",
        f"用 v4 的 IsaacLab/PhysX 状态与接触回放训练世界模型，复用冻结的 v4 Flow 和 Joint-FM。"
        f"按采集种子拆分为 {split[0]}、{split[1]}、{split[2]}。"
        f"新规划器使用 {loop['history']} 帧状态历史和 24 步预测，在线按推断的回弹后验对三个模态风险加权。"
        "单步风险阈值固定为 18 mm，终端外推警戒阈值独立固定为 90 mm、权重 3200。"
        "旧的验证集净空分位数仅保留为数据统计，不再覆写风险阈值。"
        "接触判定保持单物理步 0.004 N·s。",
        "",
        "### 24 帧遮挡记忆",
        "",
        "从每次真实交接取前缀中的 6 帧盖角和角速度，随后拼接 18 个带可见性标记的遮挡帧。"
        "同时遮掉盖角，避免从位置差分恢复速度。训练对可见帧加噪，使用三个初始化种子。"
        "轨迹预测使用记录中交接后的实际动作与标签，遮挡为离线输入干预，物理系统未额外等待 18 步。"
        "表中准确率给出跨初始化均值与总体标准差，各模型使用相同的 44 个留出交接。",
        "",
        table(["历史编码器", "参数", "留出样本/种子", "辨识准确率", "盖角 MAE rad", "角速度 MAE rad/s"], mem_rows),
        "",
        "### 世界模型维度",
        "",
        f"各维度共用 128 维历史编码器架构，用同一初始化种子独立训练 {dimensions['128']['updates']} 次更新。"
        "4 维对照直接递推净空、占用、进度和冲量的四个安全坐标，32/128/256 维使用 Mamba-Future 与独立读出。"
        f"预测结果来自同一组 {test['windows']} 个重叠留出窗口，漏报按窗口内逐步计算。"
        "通用状态读出有盖角、盖速、双臂末端位置及物体坐标监督，遮挡时的物体坐标从损失与误差中屏蔽。",
        "",
        table(["D_z", "WM 参数", "净空 MAE mm", "盖角 MAE rad", "盖速 MAE rad/s", "可见物体 MAE mm",
               "碰撞漏报/正例", "误报/负例", "闭环成功", "闭环夹碰"], dim_rows),
        "",
        "### 固定动作候选排序",
        "",
        "在留出回放中取同种子同回弹档的 direct 与 hold，预测时两条动作使用同一个 direct 交接观测。"
        f"仿真独立采样的实际前缀最大特征差为 {prefix_gap:.3f}（原始各通道单位混合），"
        "安全排序要求 static 选 direct、slow/fast 选 hold，且所选记录分支峰值冲量低于 0.004 N·s；保留此前缀差异。软风险代价自身也可能偏好有碰撞的分支，因此不以预测/标签代价排序一致代替安全排序。",
        "",
        table(["D_z", "安全物理分支排序通过", "种子/回弹档 · 预测/安全分支"], ranking),
        "",
        "各规划器另从相同留出观测生成动作，每个交接使用四个共用的噪声种子。"
        "按前 8 步双臂位置平均距离，把生成动作归入最近的记录候选，统计它是否对应 static→direct、slow/fast→hold 的安全物理分支。"
        "这是候选偏好代理指标，两个已执行分支才有真实物理风险标签。",
        "",
        table(["方法", "总体吻合", "static", "slow", "fast"], planner_rows),
        "",
        "### 常规闭环",
        "",
        f"仿真种子 {', '.join(map(str, seeds))}。当 fast 后验超过 {loop['fast_probability']}，或盖子向闭合方向的速度"
        f"超过 {loop['closing_rad_s']} rad/s，执行 4 步后重规划，否则执行 8 步。"
        "所有自适应方法使用同类调度，BC 与 Joint-FM 的计时含用于调度的 Mamba-Past 后验。"
        "两件物品均完成装盒且盖角小于 0.075 rad 才计为成功。仿真在规划期间暂停物理推进。",
        "",
        table(["方法", "static 成功", "static 夹碰", "slow 成功", "slow 夹碰", "fast 成功", "fast 夹碰",
               "4 步窗口/决策", "延迟 P50 ms"], main_rows),
        "",
        f"自适应 Ours 装盒 {count(ours, 'success')}、夹碰 {count(ours, 'pinch_or_collision')}。"
        f"固定 8 步 Ours 装盒 {count(fixed, 'success')}、夹碰 {count(fixed, 'pinch_or_collision')}。",
        "",
        "### 半落扰动恢复",
        "",
        "第一件物品进入前注入预设松手和探入扰动。安全恢复要求右臂实际撤回、随后盒盖重开、"
        "最终完成装盒且全回合无夹碰。最后一列统计任一物品首次规划之前的预释放或扰动夹碰。",
        "",
        table(["方法", "回弹档", "装盒", "撤回并重开", "安全恢复", "夹碰", "规划前已有夹碰"], recovery_rows),
        "",
        f"两档扰动合并后，自适应 Ours 安全恢复 {count(ours_recovery, 'safe_recovery')}，"
        f"固定 8 步为 {count(fixed_recovery, 'safe_recovery')}，BC 为 {count(bc_recovery, 'safe_recovery')}。",
        "",
        "Isaac Sim 5.1 以 headless 模式运行，使用既有 ALOHA USD 和 PhysX。日志含资产可视化引用未解析警告。"
        "闭环记录来自仿真关节、几何状态和接触传感器。"
        + ("同机按种子分片运行，各分片的 GPU 策略决策延迟含训练及其他分片的并行负载。" if "shard_sizes" in loop else ""),
        "",
    ]
    return "\n".join(lines), normal


def tau_report(push, spring, seeds, timing=None):
    out, push_points, spring_points = [], [], []
    for tau in TAUS:
        p = push[{0.0: "Early In-Flow", 0.7: "Ours"}.get(tau, f"tau={tau}")]
        s = group(spring, "variant", "Ours" if tau == 0.7 else f"tau={tau}")
        if {(row["regime"], row["seed"]) for row in s} != {(r, seed) for r in REGIMES for seed in seeds}:
            raise ValueError(f"Incomplete spring tau={tau}")
        safe = sum(row["success"] and not row["pinch_or_collision"] for row in s)
        method = {0.0: "Early In-Flow", 0.7: "Ours"}.get(tau, f"tau={tau}")
        push_ms = timing["summary"][method]["p50_ms"] if timing else np.median(push_latency(p))
        push_points.append((average(p, "final_iou"), push_ms))
        spring_points.append((safe, np.median(packed_samples(s, "ms"))))
        out.append([f"{tau:.1f}", f'{average(p, "final_iou"):.3f}', count(p, "success"),
                    f'{push_ms:.1f}', count(s, "success"), count(s, "pinch_or_collision"), f"{safe}/{len(s)}",
                    f'{packed_samples(s, "world_rollouts").mean():.1f}', f'{np.median(packed_samples(s, "ms")):.1f}'])
    for points in (push_points, spring_points):
        for row, (score, latency) in zip(out, points, strict=True):
            dominated = any(a >= score and b <= latency and (a > score or b < latency) for a, b in points)
            row.append("—" if dominated else "是")
    return "\n".join([
        "## 介入时机与耗时",
        "",
        "各 τ 使用 K=8。τ=1.0 完整生成后排序，τ=0.0 在初始噪声一步外推后就开始筛选和引导。"
        "中间各档使用相同欧拉网格、同一损失和精确端点雅可比。"
        "前沿分别按最终 IoU 或无夹碰装盒数与决策 P50 延迟计算。"
        + ("Push-T 此表使用全部并行作业结束后的独立计时，固定相同留出历史、轮换方法顺序，每方法 20 次。" if timing else ""),
        "",
        table(["τ_start", "Push-T 最终 IoU", "Push-T 成功", "Push-T P50 ms", "弹簧盖装盒", "弹簧盖夹碰",
               "无夹碰装盒", "弹簧盖 WM 轨迹/决策", "弹簧盖 P50 ms", "Push-T 前沿", "弹簧盖前沿"], out),
        "",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--run-dir", default="phase2_1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "outputs"
    p = root / args.run_dir / "pusht"
    s = root / args.run_dir / "spring"
    if args.merge_shards:
        merge_shards(p, s)
    push, push_arms = push_report(read(p / "results.json"), read(p / "training.json"))
    loop = read(s / "closed_loop.json")
    spring, spring_arms = spring_report(read(s / "memory.json"), read(s / "dimensions.json"), loop,
                                        read(s / "planner_ranking.json"))
    timing = read(p / "isolated_timing.json") if (p / "isolated_timing.json").exists() else None
    tau = tau_report(push_arms, spring_arms, loop["seeds"], timing)
    links = [
        ("Push-T 数据与训练", "phase2/pusht/training.json"),
        ("Push-T 扰动回放", "phase2/pusht/exploration.npz"),
        ("Push-T 逐回合", "phase2/pusht/results.json"),
        ("遮挡记忆", "phase2/spring/memory.json"),
        ("维度及离线排序", "phase2/spring/dimensions.json"),
        ("规划器候选偏好", "phase2/spring/planner_ranking.json"),
        ("弹簧盖逐回合", "phase2/spring/closed_loop.json"),
        ("验证集风险排序", "phase2/spring/validation_ranking.json"),
        ("独立计时", "phase2/pusht/isolated_timing.json"),
        ("逐项验收", "phase2/acceptance.json"),
        ("v4 备份", "Phase1_实验报告_v4.md"),
        ("v4 评测备份", "evaluation/results_v4.md"),
    ]
    links = [(label, path.replace("phase2/", args.run_dir + "/")) for label, path in links]
    def sources(prefix):
        return " · ".join(f"[{label}]({prefix}{path})" for label, path in links)

    architecture = (
        "本轮使用本机 RTX 5060 的 PyTorch 2.9.0+cu128 完成训练与策略推理。"
        "Push-T 使用 256 通道条件时序 ResNet 直接预测速度场，Flow 批量 256、余弦退火；世界模型批量 128。"
        "Pymunk 重置先赋角度再赋几何位置。弹簧盖风险阈值为 18/90 mm，梯度计算内部移除动作步长硬投影，输出仍受步长约束。"
        "本轮 Push-T 最长 300 步，首轮为 200 步；跨轮变化包含模型、训练、风险目标及预算变化，不能归因于单项修复。"

        "统一世界模型用因果深度卷积与原生 PyTorch 选择性扫描处理历史和未来。"
        "对角递推采用 exp(ΔA) 与规范给定的 ΔB 输入近似，Mamba-Future 逐帧融合上一隐状态与候选动作。"
        "状态重构与安全读出分别监督。GPC 在 τ=1.0 后排序或做投影梯度下降，"
        "Ours 在 τ=0.7 用一步外推排序，再在最后三个积分步通过端点的完整雅可比引导。"
    )
    report = "\n\n".join(["# Phase 2.1 · GPU 与风险通路修复后的实测报告", architecture, push, spring, tau, sources("")])
    phase1 = "\n\n".join(["# 弹簧盖实验报告 · Phase 2.1", architecture, spring, tau,
                           "[完整 Phase 2 报告](Phase2_GPC_Mamba_实验报告.md) · " + sources("")])
    results = "\n\n".join(["# Phase 2.1 评测结果", push, spring, tau,
                            "[完整 Phase 2 报告](../Phase2_GPC_Mamba_实验报告.md) · " + sources("../")])
    for name, content in (("Phase2_GPC_Mamba_实验报告.md", report), ("Phase1_实验报告.md", phase1),
                          ("evaluation/results.md", results)):
        path = root / name
        tmp = path.with_suffix(".tmp")
        tmp.write_text(content.rstrip() + "\n", encoding="utf-8")
        tmp.replace(path)


if __name__ == "__main__":
    main()
