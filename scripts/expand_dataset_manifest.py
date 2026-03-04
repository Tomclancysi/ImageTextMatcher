import argparse
import csv
import random
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests


DEFAULT_PREFIX = "https://modelscope.cn-beijing.oss.aliyuncs.com/open_data/sa-1b-cot-qwen/"


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_existing_ids(rows: List[Dict[str, str]]) -> Tuple[Set[int], Optional[int], Optional[int], str]:
    ids: Set[int] = set()
    prefix = DEFAULT_PREFIX

    for row in rows:
        url = (row.get("url") or "").strip()
        if not url:
            continue

        file_name = url.rsplit("/", 1)[-1]
        if file_name.startswith("sa_") and file_name.endswith(".jpg"):
            try:
                ids.add(int(file_name[3:-4]))
            except ValueError:
                continue

        if "/sa_" in url:
            prefix = url.rsplit("/", 1)[0] + "/"

    if not ids:
        return ids, None, None, prefix

    return ids, min(ids), max(ids), prefix


def build_session(timeout: int) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "image/*,*/*;q=0.8",
        }
    )
    session.request_timeout = timeout
    return session


def probe_image(session: requests.Session, prefix: str, image_id: int) -> bool:
    url = f"{prefix}sa_{image_id}.jpg"
    try:
        response = session.head(url, timeout=session.request_timeout, allow_redirects=True)
        if response.status_code != 200:
            return False
        content_type = (response.headers.get("content-type") or "").lower()
        return "image/" in content_type or content_type == "application/octet-stream"
    except requests.RequestException:
        return False


def discover_ids(
    existing_ids: Set[int],
    target_total: int,
    prefix: str,
    min_id: int,
    max_id: int,
    timeout: int,
    workers: int,
    seed: int,
    max_attempts: int,
) -> List[int]:
    additional_needed = max(0, target_total - len(existing_ids))
    if additional_needed == 0:
        return []

    rng = random.Random(seed)
    scheduled: Set[int] = set()
    found: List[int] = []
    session = build_session(timeout)

    def next_candidate() -> Optional[int]:
        for _ in range(1000):
            candidate = rng.randint(min_id, max_id)
            if candidate in existing_ids or candidate in scheduled:
                continue
            scheduled.add(candidate)
            return candidate
        return None

    attempts = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        initial = min(workers, max_attempts)
        for _ in range(initial):
            candidate = next_candidate()
            if candidate is None:
                break
            futures[pool.submit(probe_image, session, prefix, candidate)] = candidate
            attempts += 1

        while futures and len(found) < additional_needed and attempts <= max_attempts:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)

            for future in done:
                candidate = futures.pop(future)
                if future.result():
                    found.append(candidate)

            while len(futures) < workers and len(found) < additional_needed and attempts < max_attempts:
                candidate = next_candidate()
                if candidate is None:
                    break
                futures[pool.submit(probe_image, session, prefix, candidate)] = candidate
                attempts += 1

    return found


def write_rows(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["url", "cap_seg"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand dataset manifest by probing additional SA image URLs.")
    parser.add_argument("--input", default="data/dataset_en.csv", help="Base CSV manifest.")
    parser.add_argument("--output", default="data/dataset_5000.csv", help="Expanded CSV manifest.")
    parser.add_argument("--target", type=int, default=5000, help="Target number of rows in the output manifest.")
    parser.add_argument("--min-id", type=int, help="Minimum SA image id to probe.")
    parser.add_argument("--max-id", type=int, help="Maximum SA image id to probe.")
    parser.add_argument("--timeout", type=int, default=10, help="HEAD request timeout in seconds.")
    parser.add_argument("--workers", type=int, default=24, help="Concurrent probe workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=50000,
        help="Maximum ids to probe before giving up.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = load_rows(input_path)
    existing_ids, inferred_min, inferred_max, prefix = parse_existing_ids(rows)

    if inferred_min is None or inferred_max is None:
        raise SystemExit("Could not infer id range from the input CSV.")

    min_id = args.min_id if args.min_id is not None else inferred_min
    max_id = args.max_id if args.max_id is not None else inferred_max

    if min_id >= max_id:
        raise SystemExit("min-id must be smaller than max-id.")

    print(f"Loaded {len(rows)} rows from {input_path}")
    print(f"Using prefix: {prefix}")
    print(f"Probe range: {min_id} - {max_id}")
    print(f"Target rows: {args.target}")

    discovered_ids = discover_ids(
        existing_ids=existing_ids,
        target_total=args.target,
        prefix=prefix,
        min_id=min_id,
        max_id=max_id,
        timeout=args.timeout,
        workers=max(1, args.workers),
        seed=args.seed,
        max_attempts=max(1, args.max_attempts),
    )

    merged_rows = list(rows)
    for image_id in discovered_ids:
        merged_rows.append(
            {
                "url": f"{prefix}sa_{image_id}.jpg",
                "cap_seg": "",
            }
        )

    write_rows(merged_rows, output_path)
    print(f"Discovered {len(discovered_ids)} new rows")
    print(f"Wrote {len(merged_rows)} rows to {output_path}")

    if len(merged_rows) < args.target:
        print("Warning: target not reached. Increase --max-attempts or widen the probe range.")


if __name__ == "__main__":
    main()
