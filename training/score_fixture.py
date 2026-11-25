"""Score one blind physical fixture run using the Android trace CSV and labels."""

import argparse
import csv
import json
import math
from pathlib import Path


def read_csv(path, required):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing {', '.join(sorted(missing))}")
        return list(reader)


def estimate(successes, trials):
    if not trials:
        return {"count": successes, "trials": 0, "fraction": None, "wilson95": None}
    p = successes / trials
    z = 1.96
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return {"count": successes, "trials": trials, "fraction": p,
            "wilson95": [max(0, center - margin), min(1, center + margin)]}


def score(labels, trace):
    truth = {}
    for row in labels:
        part, label = row["part"].strip(), row["truth"].strip().lower()
        if not part or part in truth or label not in {"clean", "defect"}:
            raise ValueError(f"Expected one clean/defect label per physical part. Invalid row: {row}")
        truth[part] = label
    if not truth:
        raise ValueError("The label sheet is empty.")
    profiles = {row["vision_model"] for row in trace if row["source"] == "REAR CAMERA" and row["part"] in truth}
    if len(profiles) != 1:
        raise ValueError("Export one live test run using a single vision profile.")
    decisions = {}
    for row in trace:
        part, decision = row["part"], row["decision"]
        if row["source"] != "REAR CAMERA" or part not in truth:
            continue
        if part in decisions or decision not in {"HOLD", "REVIEW", "PASS"}:
            raise ValueError(f"Expected one terminal decision for {part}. Duplicate or invalid decision: {decision}")
        decisions[part] = decision
    buckets = {}
    for label in ("defect", "clean"):
        observed = [decisions.get(part, "missing") for part in truth if truth[part] == label]
        buckets[label] = {decision: observed.count(decision) for decision in ("HOLD", "REVIEW", "PASS", "missing")}
    defects, clean = buckets["defect"], buckets["clean"]
    return {
        "scope": "One terminal decision per labeled physical part. Missing decisions remain in the study denominator.",
        "profile": profiles.pop(),
        "counts": buckets,
        "defects_held": estimate(defects["HOLD"], sum(defects.values())),
        "defects_held_or_reviewed": estimate(defects["HOLD"] + defects["REVIEW"], sum(defects.values())),
        "clean_false_holds": estimate(clean["HOLD"], sum(clean.values())),
        "clean_holds_or_reviews": estimate(clean["HOLD"] + clean["REVIEW"], sum(clean.values())),
        "missing": [part for part in truth if part not in decisions],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path, help="CSV with part,truth and truth equal to clean or defect")
    parser.add_argument("trace", type=Path, help="CSV exported by the Android app")
    args = parser.parse_args()
    try:
        labels = read_csv(args.labels, ("part", "truth"))
        trace = read_csv(args.trace, ("part", "source", "decision", "vision_model"))
        print(json.dumps(score(labels, trace), indent=2))
    except (OSError,ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
