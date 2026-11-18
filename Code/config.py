import os

# ─── API Settings ───
# Cloud API proxy (default):
#   BASE_URL = "YOUR_API_BASE_URL"
# Local vLLM instance:
#   BASE_URL = "http://localhost:8000/v1"
#   API_KEY  = "no"
# Override at runtime via env vars:
#   OPENAI_API_BASE=http://localhost:8000/v1 OPENAI_API_KEY=no python eval_foresight.py ...
# Or use the --local flag in eval_foresight.py (recommended).
BASE_URL = os.environ.get("OPENAI_API_BASE", "YOUR_API_BASE_URL")
API_KEY = os.environ.get("OPENAI_API_KEY", None) or "YOUR_API_KEY"

# ─── Model Settings ───
def _find_model(name, *candidates):
    """Find model path: check env var first, then candidate directories."""
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]  # fallback to first candidate

# Phase 1: Local BLIP-2 for captioning (runs on GPU)
# DEPRECATED: Phase 1 now uses the unified VLM (VERIFY_MODEL) instead of BLIP-2.
# These settings are kept for backward compatibility with foresight_agent_api.py only.
BLIP2_MODEL_PATH = os.environ.get("BLIP2_MODEL_PATH") or _find_model(
    "blip2-opt-2.7b",
    "/data/blip2-opt-2.7b",
    "/workspace/user_code/models/blip2-opt-2.7b",
)
# Phase 1: Local CLIP for scene boundary detection (runs on GPU)
# DEPRECATED: Scene boundaries now detected by unified VLM in Phase 1.
CLIP_MODEL_PATH = os.environ.get("CLIP_MODEL_PATH") or _find_model(
    "clip-vit-base-patch32",
    "/data/clip-vit-base-patch32",
    "/workspace/user_code/models/clip-vit-base-patch32",
)
# Phase 2/2.5: LLM for structure reasoning (text-only).
#   Cloud default : "gemini-3.1-pro"
#   Local vLLM    : auto-detected from /v1/models, or set STRUCTURE_MODEL explicitly.
STRUCTURE_MODEL = os.environ.get("STRUCTURE_MODEL", "gemini-3.1-pro")
# Phase 1 + Phase 2.5 + Phase 3: VLM for visual understanding.
#   Phase 1 : multi-image captioning + scene boundary detection
#   Phase 2.5: single-frame probe captioning
#   Phase 3  : targeted answer verification
#   Cloud default : "qwen3.5-35b-a3b"
#   Local vLLM    : auto-detected from /v1/models, or set VERIFY_MODEL explicitly.
#   Minimum viable: Qwen2.5-VL-7B-Instruct  (lower bound test)
VERIFY_MODEL = os.environ.get("VERIFY_MODEL", "qwen3.5-35b-a3b")
# Caption model label — used for logging only
CAPTION_MODEL = os.environ.get("VERIFY_MODEL", "qwen3.5-35b-a3b")  # same as VERIFY_MODEL

# ─── Dataset Settings ───
# DATA_DIR: directory that contains all JSONL data files.
# Default is a "data/" subdirectory relative to the working directory.
# Override: DATA_DIR=/group/.../data  python eval_foresight.py ...
#       or: DATA_DIR=/group/.../data  bash run_gpu.sh
_DATA_DIR = os.environ.get("DATA_DIR", "data")

def _j(filename):
    """Join DATA_DIR with a filename."""
    return os.path.join(_DATA_DIR, filename)

DATASET_CONFIG = {
    "videomme_test": {
        "jsonl": _j("videomme_test_5videos.jsonl"),
        "video_root": os.environ.get(
            "VIDEO_ROOT",
            "/path/to/aaai/eva/our_project/video_data/videos",
        ),
        "cache": "cache_videomme_test.jsonl",
        "result": "result_videomme_test.jsonl",
    },
    "videomme_33": {
        "jsonl": _j("videomme_test_33videos.jsonl"),
        "video_root": os.environ.get(
            "VIDEO_ROOT",
            "/path/to/aaai/eva/our_project/video_data/videos",
        ),
        "cache": "cache_videomme_33.jsonl",
        "result": "result_videomme_33.jsonl",
    },
    "videomme_balanced": {
        "jsonl": _j("videomme_balanced_types.jsonl"),
        "video_root": os.environ.get(
            "VIDEO_ROOT",
            "/path/to/aaai/eva/our_project/video_data/videos",
        ),
        "cache": "cache_videomme_balanced.jsonl",
        "result": "result_videomme_balanced.jsonl",
    },
    "videomme_supp": {
        "jsonl": _j("videomme_supplementary.jsonl"),
        "video_root": os.environ.get(
            "VIDEO_ROOT",
            "/path/to/aaai/eva/our_project/video_data/videos",
        ),
        "cache": "cache_videomme_supp.jsonl",
        "result": "result_videomme_supp.jsonl",
    },
    "videomme": {
        "jsonl": _j("videomme_test_wosubtitles_raw_list_full.jsonl"),
        "video_root": os.environ.get(
            "VIDEO_ROOT",
            "/path/to/aaai/eva/our_project/video_data/videos",
        ),
        "cache": "cache_videomme.jsonl",
        "result": "result_videomme.jsonl",
    },
    "longvideobench": {
        "jsonl": _j("LongVideoBench_nosub_raw_full.jsonl"),
        "video_root": os.environ.get("VIDEO_ROOT_LVB", "/path/to/datasets/LongVideoBench/videos"),
        "cache": "cache_longvideobench.jsonl",
        "result": "result_longvideobench.jsonl",
    },
    "lvbench": {
        "jsonl": _j("LVBench_raw_full_root.jsonl"),
        "video_root": os.environ.get("VIDEO_ROOT_LVBENCH", "/path/to/datasets/LVBench/videos"),
        "cache": "cache_lvbench.jsonl",
        "result": "result_lvbench.jsonl",
    },
    "videoholmes": {
        "jsonl": _j("videoholmes_raw_full.jsonl"),
        "video_root": os.environ.get("VIDEO_ROOT_HOLMES", "/path/to/datasets/Video-Holmes/videos"),
        "cache": "cache_videoholmes.jsonl",
        "result": "result_videoholmes.jsonl",
    },
    "videommu": {
        "jsonl": _j("VideoMMMU_raw_full.jsonl"),
        "video_root": os.environ.get("VIDEO_ROOT_MMU", "/path/to/datasets/VideoMMMU/videos"),
        "cache": "cache_videommu.jsonl",
        "result": "result_videommu.jsonl",
    },
    "mlvu": {
        "jsonl": _j("MLVU_MCQ_raw_full.jsonl"),
        "video_root": os.environ.get("VIDEO_ROOT_MLVU", "/path/to/datasets/MLVU/MLVU"),
        "cache": "cache_mlvu.jsonl",
        "result": "result_mlvu.jsonl",
    },
    # RL training data converted to eval format (absolute paths in videos[0], video_root unused)
    "train_grpo": {
        "jsonl": os.environ.get(
            "TRAIN_GRPO_JSONL",
            "/path/to/datasets/CG-Bench/grpo_parquet_multi/train_eval.jsonl",
        ),
        "video_root": "/",   # absolute paths in JSONL override this
        "cache": "cache_train_grpo.jsonl",
        "result": "result_train_grpo.jsonl",
    },
}

# ─── Phase 1: Sparse Captioning ───
NUM_PROBE_FRAMES = int(os.environ.get("NUM_PROBE_FRAMES", "5"))  # Number of initial probe frames (ablation: 3-6)
PHASE1_RESIZE = 0.5         # Resolution factor for Phase 1 frame extraction
FRAME_SAVE_ROOT = "extracted_frames"
CLIP_BOUNDARY_THRESHOLD = 0.3  # Legacy — kept for reference only

# ─── Phase 2.5: Adaptive Probing ───
MAX_PROBE_ROUNDS = int(os.environ.get("MAX_PROBE_ROUNDS", "4"))  # How many adaptive probing iterations
MAX_EXTRA_FRAMES = int(os.environ.get("MAX_EXTRA_FRAMES", "3"))  # Max extra frames per probing round
CAPTION_GAP_THRESHOLD = 20  # Seconds: adjacent caption gap > this triggers probing
PROBE_DEDUP_DIST = 5        # Seconds: merge probe candidates closer than this

# ─── Phase 3: Retrieval-Free Verification ───
MAX_VERIFY_ROUNDS = 5       # Max VLM verification rounds
DEFAULT_VERIFY_FRAMES = 10  # Default frames for verification region
CONFIDENCE_THRESHOLD = 3    # Confidence >= this → stop (1=insufficient/2=partial/3=sufficient)
MAX_VISUAL_TOKENS = 24000   # Visual token budget per VLM call (raised for v9j full-quality frames)
MAXP = 720                  # Max resolution for fallback

# ─── Evaluation ───
MAX_CONCURRENT = 10         # Concurrency limit
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))  # 0.0=deterministic; set >0 for pass@K rollouts
# Max tokens for Phase 1+2 merged VLM output (captions/boundaries/structure/gap_inference/query_reasoning).
# Bumped to 1800 by default because the world-model prompts (v9e / v9f) produce notably longer outputs
# (gap_inference + per-segment reasoning). 1536 used to truncate ~10-15% of long-video samples → parse
# failure → uniform-probe fallback → silently disables the world model. Lower it via env var for older
# prompts to save tokens.
MAX_TOKENS_PHASE1 = int(os.environ.get("MAX_TOKENS_PHASE1", "1800"))
MAX_TOKENS_STRUCTURE = 2048 # Max tokens for Phase 2 LLM output
MAX_TOKENS_VERIFY = int(os.environ.get("MAX_TOKENS_VERIFY", "1500"))    # Max tokens for Phase 3 VLM output (raised for <conclusion> tag)
