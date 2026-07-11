import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to jsonl file")
    parser.add_argument(
        "--sample", type=int, default=5, help="How many example line numbers to show"
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    total_lines = 0
    non_empty_lines = 0
    json_ok = 0
    json_fail = 0
    empty_line_nums = []
    json_fail_nums = []
    doc_id_counts = Counter()

    with input_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            total_lines += 1
            raw = line.strip()
            if not raw:
                if len(empty_line_nums) < args.sample:
                    empty_line_nums.append(idx)
                continue
            non_empty_lines += 1
            try:
                obj = json.loads(raw)
                json_ok += 1
                doc_id = obj.get("doc_id")
                if doc_id is not None:
                    doc_id_counts[doc_id] += 1
            except json.JSONDecodeError:
                json_fail += 1
                if len(json_fail_nums) < args.sample:
                    json_fail_nums.append(idx)

    dup_doc_ids = {k: v for k, v in doc_id_counts.items() if v > 1}

    print(f"file: {input_path}")
    print(f"total_lines: {total_lines}")
    print(f"non_empty_lines: {non_empty_lines}")
    print(f"json_ok: {json_ok}")
    print(f"json_fail: {json_fail}")
    if empty_line_nums:
        print(f"empty_line_samples: {empty_line_nums}")
    if json_fail_nums:
        print(f"json_fail_samples: {json_fail_nums}")
    if dup_doc_ids:
        top_dups = sorted(dup_doc_ids.items(), key=lambda x: -x[1])[: args.sample]
        print(f"duplicate_doc_id_samples: {top_dups}")


if __name__ == "__main__":
    main()
