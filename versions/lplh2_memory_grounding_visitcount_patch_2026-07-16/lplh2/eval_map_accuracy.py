"""Offline comparison of text-built room labels with Jericho evaluation ids."""

from __future__ import annotations

import argparse
import collections
import json


def evaluate(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    pairs = []
    for epoch in payload.get("epochs", []):
        for step in epoch.get("steps", []):
            room_num = step.get("eval_room_num")
            label = (
                step.get("modules", {}).get("kg_map", {}).get("current_location")
                or step.get("location")
            )
            if room_num is not None and label:
                pairs.append((str(label), int(room_num)))

    label_counts = collections.defaultdict(collections.Counter)
    room_counts = collections.defaultdict(collections.Counter)
    for label, room_num in pairs:
        label_counts[label][room_num] += 1
        room_counts[room_num][label] += 1

    majority = {
        label: counts.most_common(1)[0][0]
        for label, counts in label_counts.items()
    }
    correct = sum(1 for label, room_num in pairs if majority[label] == room_num)
    merges = {
        label: dict(counts)
        for label, counts in label_counts.items()
        if len(counts) >= 2
    }
    splits = {
        str(room_num): dict(counts)
        for room_num, counts in room_counts.items()
        if len(counts) >= 2
    }
    return {
        "paired_steps": len(pairs),
        "step_identity_accuracy": (correct / len(pairs)) if pairs else 0.0,
        "merge_errors": merges,
        "split_errors": splits,
        "per_engine_room": {
            str(room_num): {
                "steps": sum(counts.values()),
                "agent_labels": dict(counts),
                "majority_label": counts.most_common(1)[0][0],
            }
            for room_num, counts in sorted(room_counts.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("steplog", help="Path to steplog.json")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.steplog), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
