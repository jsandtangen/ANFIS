"""Recreate the cost audit from the saved class-weighted test confusion counts."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

OUT = Path(__file__).resolve().parent / "results" / "research"
RATIOS = (1, 2, 5)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    per_split = []
    groups = defaultdict(list)
    with (OUT / "per_split_metrics.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["condition"] != "weighted":
                continue
            counts = {key: int(row[key]) for key in ("tn", "fp", "fn", "tp")}
            n = sum(counts.values())
            for ratio in RATIOS:
                cost = (counts["fp"] + ratio * counts["fn"]) / n
                record = dict(seed=int(row["seed"]), model=row["model"],
                              threshold_mode=row["threshold_mode"],
                              threshold=float(row["threshold"]), **counts,
                              n=n, fn_to_fp=ratio, cost=cost)
                per_split.append(record)
                groups[row["model"], row["threshold_mode"], ratio].append(record)
    summary = []
    for (model, mode, ratio), records in sorted(groups.items()):
        seeds = {row["seed"] for row in records}
        if len(records) != 5 or seeds != {42, 7, 19, 73, 101}:
            raise ValueError(f"Unexpected seeds for {model}, {mode}, {ratio}: {seeds}")
        costs = [row["cost"] for row in records]
        summary.append(dict(model=model, threshold_mode=mode, fn_to_fp=ratio,
                            splits=len(costs), mean_cost=mean(costs),
                            sample_sd=stdev(costs)))
    write_csv(OUT / "ACIT4620_cost_per_split.csv", per_split)
    write_csv(OUT / "ACIT4620_cost_summary.csv", summary)


if __name__ == "__main__":
    main()
