"""Markdown tables for individual workers and the merged evaluation."""

from statistics import mean


REGIMES = ("static", "slow", "fast")
METHODS = ("M1", "M2", "M3", "M4-a", "M4-b", "B1", "B2")
ABLATIONS = (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)
RECOVERY_METHODS = ("M1", "M3", "M4-b")


def table_row(prefix, rows):
    chunks = [chunk for row in rows for chunk in row["chunks"]]
    elapsed, engaged, cost = (mean(row[key] for row in rows) for key in ("elapsed_s", "left_engaged_s", "cost"))
    success = 100 * mean(row["success"] for row in rows)
    collision = 100 * mean(row["pinch_or_collision"] for row in rows)
    ms = f'{mean(s["ms"] for s in chunks):.2f}' if chunks else "—"
    calls = f'{mean(s["world_batches"] for s in chunks):.2f}' if chunks else "—"
    return f"| {prefix} | {len(rows)} | {success:.1f} | {collision:.1f} | {elapsed:.2f} | {engaged:.2f} | {cost:.2f} | {ms} | {calls} |"


def write_report(out, rows, ablation, recovery=()):
    cols = "n | 成功率 % | 夹碰率 % | 仿真秒 | 左手扶盖秒 | 回合代价 | ms/块 | WM 批调用/块"
    seeds = "、".join(str(seed) for seed in sorted({row["seed"] for row in rows}))
    report = [
        "# Phase 1 台架评测", "",
        f"评估种子为 {seeds}。CPU IsaacLab / PhysX。开盖、抓取和关盖共用脚本，比较方法控制插入和恢复。"
        "回合代价为仿真秒数加 0.35 倍左手占用秒数。B1/B2 采用规则控制。", "",
        f"| 方法 | 回弹档 | {cols} |", "|---|---|" + "---|" * 8,
    ]
    for method in METHODS:
        for regime in REGIMES:
            matching = [row for row in rows if row["method"] == method and row["regime"] == regime]
            if matching:
                report.append(table_row(f"{method} | {regime}", matching))
    report += ["", "## τ_start 消融", "", f"| τ_start | {cols} |", "|---|" + "---|" * 8]
    for tau in ABLATIONS:
        matching = [row for row in ablation if row["tau_start"] == tau]
        if matching:
            report.append(table_row(f"{tau:.1f}", matching))
    if recovery:
        report += ["", "## Perturbed Recovery", "",
                   "仅在第一件交接前注入预回落和探入。无夹碰恢复要求实测右手外撤、随后盒盖重开、最终完整装盒且整回合无夹碰。", "",
                   "| 方法 | 回弹档 | n | 装盒成功 | 撤回并重开 | 无夹碰恢复 | 夹碰 |",
                   "|---|---|---:|---:|---:|---:|---:|"]
        for method in RECOVERY_METHODS:
            for regime in ("slow", "fast"):
                matching = [r for r in recovery if r["method"] == method and r["regime"] == regime]
                if matching:
                    n = len(matching)
                    counts = (sum(r[k] for r in matching) for k in (
                        "success", "withdrew_and_reopened", "safe_recovery", "pinch_or_collision",
                    ))
                    report.append(f"| {method} | {regime} | {n} | " + " | ".join(f"{v}/{n}" for v in counts) + " |")
    (out / "results.md").write_text("\n".join(report) + "\n", encoding="utf-8")
