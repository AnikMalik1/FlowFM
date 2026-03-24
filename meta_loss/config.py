"""
Meta-Loss Evolution Framework — Configuration
"""
import os

# ── LLM API (OpenAI-compatible) ─────────────────────────
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:18045/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "sk-c95e13f4ccea4d91ae24e4c132fd6971")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.1-pro-high")

# ── Paths ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FLOWFM_DIR = os.path.join(os.path.dirname(BASE_DIR), "FlowFM")
FLOWFM_REPO_DIR = os.path.join(os.path.dirname(BASE_DIR), "FlowFM_repo")
DATA_DIR = os.path.join(FLOWFM_DIR, "data") + "/"  # FinGAN.py concatenates dataloc+ticker+".csv"
ETF_LIST = os.path.join(FLOWFM_REPO_DIR, "stocks-etfs-list.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOSSES_DIR = os.path.join(BASE_DIR, "losses")

# ── Stage 1: Fast Screen ───────────────────────────────
STAGE1_TICKERS = ["AMZN", "BLK", "APA"]
STAGE1_MAX_EPOCHS = 200
STAGE1_EVAL_EVERY = 20
STAGE1_PATIENCE = 6

# ── Stage 2: Full Validation ───────────────────────────
STAGE2_TICKERS = [
    "AMZN", "HD", "NKE",
    "CL", "EL", "KO", "PEP",
    "APA", "OXY",
    "WFC", "GS", "BLK",
    "PFE", "HUM",
    "FDX", "GD",
    "IBM", "TER",
    "ECL", "IP",
    "DTE", "WEC",
    "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLU",
]
STAGE2_MAX_EPOCHS = 500
STAGE2_EVAL_EVERY = 20
STAGE2_PATIENCE = 8

# ── Model Hyperparams (FlowFMPlus defaults) ────────────
LOOKBACK = 10          # l: condition window length
HIDDEN = 256
DEPTH = 4
DROPOUT = 0.05
BATCH_SIZE = 2048
LR = 3e-4
WEIGHT_DECAY = 0.01
EMA_DECAY = 0.999
COND_NOISE_STD = 0.02
ODE_STEPS = 40

# ── Evaluation ──────────────────────────────────────────
VAL_MC_SAMPLES = 256   # MC samples for checkpoint selection
VAL_MC_CHUNK = 8
FINAL_MC_SAMPLES = 1000
FINAL_MC_CHUNK = 8

# ── Evolution ───────────────────────────────────────────
MAX_ROUNDS = 30
STAGE2_PROMOTION_TOP_K = 3  # promote to Stage 2 if SR > top-K threshold
