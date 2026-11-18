import argparse
import json
import os
import multiprocessing
from collections import Counter
from typing import Any, Dict, List

from openai import OpenAI
import tqdm

import config
# AGENT_MODULE env var lets you swap agent implementations:
#   AGENT_MODULE=foresight_agent_v9j  → unified loop (v9j)
#   (default)                         → original Phase 2.5 + Phase 3 (foresight_agent)
import importlib as _agent_imp
_agent_mod = os.environ.get("AGENT_MODULE", "foresight_agent")
_agent = _agent_imp.import_module(_agent_mod)
foresight_single = _agent.foresight_single


def get_verify_model_shortname(model_name: str) -> str:

    # Remove common suffixes like -instruct, -chat, etc.
    import re
    name = model_name.lower()
    # Remove version/size specs: 235b-a22b, 72b, etc.
    name = re.sub(r'-\d+[bm]-[a-z\d]+', '', name)
    name = re.sub(r'-\d+[bm]', '', name)
    # Remove common suffixes
    name = re.sub(r'-(instruct|chat|preview|flash)$', '', name)
    # Remove dots and dashes, keep only alphanumeric
    name = re.sub(r'[^a-z0-9]', '', name)
    return name[:20]  # Limit length


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f):
            item = json.loads(line)
            if not isinstance(item.get("index"), int):
                item["index"] = lineno
            items.append(item)
    return items


def load_cache(cache_path: str) -> Dict[int, Dict[str, Any]]:
    records = {}
    if not os.path.exists(cache_path):
        return records
    with open(cache_path, "r") as f:
        for line in reversed(f.readlines()):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = data["index"]
            if idx not in records:
                records[idx] = data
    return records


def is_normal(stop_reason):
    return stop_reason in {
        "answer_found", "low_confidence_answer", "max_rounds_exceeded",
        "no_answer", "frame_extraction_failed",
    }


def should_retry(record):
    if record.get("error"):
        return True
    return not is_normal(record.get("stop_reason"))


def process_chunk(data_chunk, video_root, output_dir, output_file, device_id=0, verbose=False):
    pid = os.getpid()
    print(f"[Worker PID-{pid}] Worker {device_id}, processing {len(data_chunk)} items")

    client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)

    for item in tqdm.tqdm(data_chunk, desc=f"GPU{device_id}", ncols=80):
        index = item["index"]
        try:
            record = foresight_single(index, item, client, video_root, output_dir, verbose=verbose)
        except Exception as e:
            record = {
                "index": index,
                "videos": item.get("videos", []),
                "gt": item.get("reward_model", {}).get("ground_truth", ""),
                "answer": "",
                "stop_reason": "exception",
                "error": f"{type(e).__name__}: {e}",
                "num_rounds": 0,
                "total_tokens": 0,
                "estimated_visual_tokens": 0,
            }

        with open(output_file, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _setup_local_vllm(local_url: str, local_model: str | None) -> str:

    from openai import OpenAI as _OpenAI

    if local_model is None:
        try:
            _probe = _OpenAI(api_key="no", base_url=local_url)
            models = _probe.models.list()
            local_model = models.data[0].id
            print(f"[Local Mode] Auto-detected model: {local_model}")
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach vLLM at {local_url} or no models loaded: {e}\n"
                "Start vLLM first, e.g.:\n"
                "  vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --max-model-len 8192"
            ) from e

    # Update config module (used by single-GPU path in main process)
    config.BASE_URL = local_url
    config.API_KEY = "no"
    config.STRUCTURE_MODEL = local_model
    config.VERIFY_MODEL = local_model

    # Set env vars so child processes inherit them on fork/spawn
    os.environ["OPENAI_API_BASE"] = local_url
    os.environ["OPENAI_API_KEY"] = "no"
    os.environ["STRUCTURE_MODEL"] = local_model
    os.environ["VERIFY_MODEL"] = local_model

    return local_model


def main():
    parser = argparse.ArgumentParser(description="ForesightAgent Evaluation")
    parser.add_argument("--dataset", required=True, choices=sorted(config.DATASET_CONFIG.keys()))
    parser.add_argument("--video-root", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs/NPUs to use in parallel")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--new-cache", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--shard-id", type=int, default=0,
        help="0-based shard index for multi-machine splitting (default: 0).",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="Total number of shards; 1 means no sharding (default: 1).",
    )
    # ── Local vLLM mode ──────────────────────────────────────────────────────
    parser.add_argument(
        "--local", action="store_true",
        help="Use a local vLLM instance instead of the cloud API.",
    )
    parser.add_argument(
        "--local-url", default="http://localhost:8000/v1", metavar="URL",
        help="Base URL of the local vLLM OpenAI-compatible server (default: %(default)s).",
    )
    parser.add_argument(
        "--local-model", default=None, metavar="MODEL",
        help=(
            "Model name served by vLLM (used for both structure and verify). "
            "If omitted, auto-detected from the server's /v1/models endpoint. "
            "Example: 'Qwen/Qwen2.5-VL-7B-Instruct'."
        ),
    )
    args = parser.parse_args()

    # ── Apply local mode overrides before any worker is spawned ──────────────
    if args.local:
        resolved = _setup_local_vllm(args.local_url, args.local_model)
        print(f"[Local Mode] URL={args.local_url}  model={resolved}")

    cfg = config.DATASET_CONFIG[args.dataset]
    video_root = args.video_root or cfg["video_root"]

    os.makedirs(args.output_dir, exist_ok=True)

    # Override result filename to include verify model short name + probe frame count
    verify_short = get_verify_model_shortname(config.VERIFY_MODEL)
    result_base, result_ext = os.path.splitext(cfg["result"])
    result_filename = f"{result_base}_{verify_short}_f{config.NUM_PROBE_FRAMES}{result_ext}"
    result_path = os.path.join(args.output_dir, result_filename)

    print(f"Models: phase1/2.5/3={config.VERIFY_MODEL}, structure={config.STRUCTURE_MODEL}")
    print(f"Probe frames: {config.NUM_PROBE_FRAMES}  API: {config.BASE_URL}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Result file: {result_path}")

    items = load_jsonl(cfg["jsonl"])
    if args.sample is not None:
        items = items[:args.sample]

    if args.num_shards > 1:
        n = len(items)
        shard_start = (n * args.shard_id) // args.num_shards
        shard_end   = (n * (args.shard_id + 1)) // args.num_shards
        items = items[shard_start:shard_end]
        shard_tag = f"_shard{args.shard_id}of{args.num_shards}"
        # Embed shard tag in result filename to avoid collisions across machines
        result_path = result_path.replace(result_ext, f"{shard_tag}{result_ext}")
        print(f"Shard: {args.shard_id}/{args.num_shards}  items [{shard_start}:{shard_end}]  ({len(items)} items)")

    cache_records = load_cache(result_path) if not args.new_cache else {}
    done_indices = {i for i, r in cache_records.items() if not should_retry(r)}
    items = [item for item in items if item["index"] not in done_indices]

    print(f"Dataset: {args.dataset}")
    print(f"Total to process: {len(items)}, Already done: {len(done_indices)}")

    if not items:
        print("Nothing to process.")
    elif args.num_gpus <= 1:
        process_chunk(items, video_root, args.output_dir, result_path, device_id=0, verbose=args.verbose)
    else:
        # Multi-GPU: split items across GPUs, each process gets its own device
        num_gpus = min(args.num_gpus, len(items))
        base_chunk, remainder = divmod(len(items), num_gpus)
        chunks = []
        start = 0
        for i in range(num_gpus):
            # Spread the remainder across the first workers instead of dumping
            # every leftover sample onto the last GPU.
            chunk_len = base_chunk + (1 if i < remainder else 0)
            end = start + chunk_len
            chunks.append(items[start:end])
            start = end

        base, ext = os.path.splitext(result_path)
        processes = []
        chunk_files = []
        for i in range(num_gpus):
            chunk_file = f"{base}_gpu{i}{ext}"
            chunk_files.append(chunk_file)
            p = multiprocessing.Process(
                target=process_chunk,
                args=(chunks[i], video_root, args.output_dir, chunk_file, i, args.verbose),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        # Merge chunk files
        with open(result_path, "a") as fout:
            for cf in chunk_files:
                if os.path.exists(cf):
                    with open(cf, "r") as fin:
                        fout.write(fin.read())
                    os.remove(cf)

    # ─── Compute accuracy ───
    all_records = load_cache(result_path)
    all_items = load_jsonl(cfg["jsonl"])
    if args.sample is not None:
        all_items = all_items[:args.sample]

    total = len(all_items)
    correct = 0
    empty = 0
    round_sum = 0.0
    vtoken_sum = 0.0       # full-pipeline visual tokens
    vtoken_p3_sum = 0.0    # Phase 3 only (legacy field)
    frame_sum = 0.0        # total frames seen across all phases
    reason_counter = Counter()

    for item in all_items:
        idx = item["index"]
        rec = all_records.get(idx)
        if rec is None:
            continue
        reason_counter[str(rec.get("stop_reason", "?"))] += 1
        round_sum += int(rec.get("num_rounds", 0))
        # Use frame_stats.total_visual_tokens when available, else fall back to estimated
        fs = rec.get("frame_stats")
        if fs:
            vtoken_sum += int(fs.get("total_visual_tokens", 0))
            frame_sum  += int(fs.get("total_frames", 0))
        else:
            vtoken_sum += int(rec.get("estimated_visual_tokens", 0))
        vtoken_p3_sum += int(rec.get("estimated_visual_tokens", 0))

        gt = str(item["reward_model"]["ground_truth"]).strip().upper()
        ans = str(rec.get("answer", "")).strip().upper()
        if not ans:
            empty += 1
            continue
        if gt == ans:
            correct += 1

    print(f"\n{'='*60}")
    print(f"Results saved to {result_path}")
    if total:
        print(f"  Accuracy: {correct}/{total} = {correct/total:.4%}")
    print(f"  Empty answers: {empty}")
    if total:
        print(f"  Avg VLM rounds: {round_sum/total:.2f}")
        print(f"  Avg visual tokens (all phases): {vtoken_sum/total:.0f}")
        print(f"  Avg visual tokens (Phase3 only): {vtoken_p3_sum/total:.0f}")
        if frame_sum > 0:
            print(f"  Avg frames seen (all phases): {frame_sum/total:.1f}")
    print(f"  Stop reasons: {dict(reason_counter.most_common())}")


if __name__ == "__main__":
    main()
