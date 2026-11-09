"""Audit the requested Phase 2.1 empirical targets without imposing outcomes."""

import json
from pathlib import Path
import numpy as np


def main():
    root = Path(__file__).resolve().parent / "outputs"
    run = root / "phase2_1"
    read = lambda name: json.loads((run / name).read_text(encoding="utf-8"))
    push, train = read("pusht/results.json"), read("pusht/training.json")
    spring, dims = read("spring/closed_loop.json"), read("spring/dimensions.json")
    timing = read("pusht/isolated_timing.json")
    checks = []

    def add(name, passed, observed):
        checks.append(dict(requirement=name, passed=bool(passed), observed=observed))

    scores = {}
    for method in push["methods"]:
        rows = [r for r in push["episodes"] if r["method"] == method]
        assert len(rows) == 50 and len({r["seed"] for r in rows}) == 50
        scores[method] = dict(iou=float(np.mean([r["final_iou"] for r in rows])),
                              success=sum(r["success"] for r in rows),
                              p50_ms=float(np.median([t for r in rows for t in r["decision_ms"]])))
    add("GPU 训练完整预算", train["device"].startswith("cuda") and 20000 <= train["flow_updates"] <= 30000
        and train["world_updates"] == 15000,
        {k: train[k] for k in ("device", "flow_updates", "world_updates")})
    bc, ours = scores["Behavior Cloning"], scores["Ours"]
    paired = {}
    baseline = {r["seed"]: r["final_iou"] for r in push["episodes"] if r["method"] == "Behavior Cloning"}
    for method in ("GPC-RANK K=16", "GPC-OPT", "GPC-RANK+OPT"):
        delta = np.array([r["final_iou"] - baseline[r["seed"]] for r in push["episodes"] if r["method"] == method])
        indices = np.random.default_rng(450).integers(0, len(delta), (10000, len(delta)))
        lo, hi = np.quantile(delta[indices].mean(1), [.025, .975])
        paired[method] = dict(mean=float(delta.mean()), ci95=[float(lo), float(hi)])
    add("三个 GPC 对照相对 BC 的配对 IoU 增益 95% 区间下界 > 0",
        all(v["ci95"][0] > 0 for v in paired.values()), paired)
    add("BC 平均最终 IoU 0.55–0.75 且成功非零", .55 <= bc["iou"] <= .75 and bc["success"] > 0, bc)
    mae = float(np.mean(train["heldout_state_mae"][2:4]) * 256)
    add("世界模型留出坐标 MAE < 5 px", mae < 5, mae)
    comparisons = ["Behavior Cloning", "Early In-Flow", "GPC-RANK K=16", "GPC-OPT"]
    add("Ours 最终 IoU 与成功数均超过指定对照",
        all(ours["iou"] > scores[m]["iou"] and ours["success"] > scores[m]["success"] for m in comparisons),
        {m: scores[m] for m in ["Ours", *comparisons]})
    ratios = {m: timing["summary"][m]["p50_ms"] / timing["summary"]["Ours"]["p50_ms"] for m in ("GPC-OPT", "GPC-RANK+OPT")}
    add("Ours 相对 OPT 与 RANK+OPT 决策 P50 至少快 4 倍", min(ratios.values()) >= 4, ratios)
    tau_names = ["Early In-Flow", "tau=0.3", "tau=0.5", "Ours", "tau=0.9", "tau=1.0"]
    values = [scores[m]["iou"] for m in tau_names]
    dominated = any(scores[m]["iou"] >= ours["iou"] and timing["summary"][m]["p50_ms"] <= timing["summary"]["Ours"]["p50_ms"]
                    and (scores[m]["iou"] > ours["iou"] or timing["summary"][m]["p50_ms"] < timing["summary"]["Ours"]["p50_ms"])
                    for m in tau_names)
    add("τ=0.7 位于 IoU–P50 延迟帕累托前沿", not dominated, dict(dominated=dominated))
    add("τ=0.7 为最终 IoU 峰值且呈倒 U 趋势",
        all(a <= b for a, b in zip(values[:3], values[1:4]))
        and all(a >= b for a, b in zip(values[3:5], values[4:]))
        and values[3] > max(values[0], values[-1]), dict(zip(tau_names, values)))
    for dim in (32, 128, 256):
        rows = dims[str(dim)]["offline_test"]
        passed = sum(r["safe_ranking_pass"] for r in rows)
        add(f"弹簧盖 {dim}D 严格安全排序 6/6", passed == 6 and len(rows) == 6, f"{passed}/{len(rows)}")
    regular = [r for r in spring["episodes"] if r["d_z"] == 128 and r["variant"] == "Ours" and not r["perturbed"]]
    assert len(regular) == 9
    collisions = sum(r["pinch_or_collision"] for r in regular)
    add("Ours 常规闭环夹碰 ≤1/9", collisions <= 1, f"{collisions}/9")
    recovery = {}
    for method in ("BC", "GPC-RANK", "GPC-OPT", "Ours"):
        rows = [r for r in spring["episodes"] if r["d_z"] == 128 and r["variant"] == method and r["perturbed"]]
        assert len(rows) == 6
        recovery[method] = sum(r["safe_recovery"] for r in rows)
    add("Ours 扰动安全恢复超过 BC/RANK/OPT", all(recovery["Ours"] > recovery[m] for m in recovery if m != "Ours"), recovery)
    result = dict(checks=checks, all_passed=all(c["passed"] for c in checks),
                  note="性能目标均为实测验收，不修改种子、分母或碰撞定义以达成目标。并行负载影响逐回合日志延迟；速度与 Push-T 前沿验收使用结束并行作业后的独立计时。")
    (run / "acceptance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["## Phase 2.1 规范目标验收", "", result["note"], "",
             "| 实测目标 | 是否达成 | 观测值 |", "| --- | --- | --- |"]
    def compact(value):
        if isinstance(value, float):
            return round(value, 4)
        if isinstance(value, dict):
            return {k: compact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [compact(v) for v in value]
        return value

    for c in checks:
        lines.append(f"| {c['requirement']} | {'通过' if c['passed'] else '未达成'} | {json.dumps(compact(c['observed']), ensure_ascii=False)} |")
    lines += ["", "未达成的结果保留为负结果；修复代码不构成安全保证或性能领先证据。", "",
              "### 独立 GPU 规划计时", "", "每方法使用同一组留出历史，预热后轮换顺序运行 20 次；计时包括采样、规划和动作传回 CPU。", "",
              "| 方法 | P50 ms | 均值 ms |", "| --- | --- | --- |"]
    for method, values in timing["summary"].items():
        lines.append(f"| {method} | {values['p50_ms']:.2f} | {values['mean_ms']:.2f} |")
    lines.append("")
    for name in ("Phase2_GPC_Mamba_实验报告.md", "Phase1_实验报告.md", "evaluation/results.md"):
        path = root / name
        existing = path.read_text(encoding="utf-8").split("## Phase 2.1 规范目标验收")[0].rstrip()
        existing = "\n".join(line for line in existing.splitlines() if not line.startswith("> Phase 2.1 实测结论："))
        title, body = existing.split("\n", 1)
        summary = (f"> Phase 2.1 实测结论：Push-T BC 最终 IoU {bc['iou']:.3f}、成功 {bc['success']}/50；"
                   f"Ours 为 {ours['iou']:.3f}、{ours['success']}/50。弹簧盖 Ours 常规装盒 "
                   f"{sum(r['success'] for r in regular)}/9、夹碰 {collisions}/9，安全扰动恢复 {recovery['Ours']}/6。"
                   f"规范的全部性能目标{'已' if result['all_passed'] else '未'}达成；详见逐项验收。")
        path.write_text(title + "\n\n" + summary + "\n" + body + "\n\n" + "\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
