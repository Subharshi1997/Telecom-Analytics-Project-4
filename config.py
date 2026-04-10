"""
Central configuration for the Telecom Analytics project.
All paths, constants, and environment settings are defined here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project Paths ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
FEATURE_STORE_DIR = ROOT_DIR / "feature_store"
MODELS_DIR = ROOT_DIR / "models_artifacts"
REPORTS_DIR = ROOT_DIR / "reports"

# Create directories if they don't exist
for d in [DATA_DIR, FEATURE_STORE_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(exist_ok=True)

# ── Raw Data Paths ────────────────────────────────────────────────────────────
RAW_DATA_PATH = Path(os.getenv(
    "RAW_DATA_PATH",
    r"C:\Users\ARKO\Downloads\telcom_data (2).xlsx"
))
FIELD_DESC_PATH = Path(os.getenv(
    "FIELD_DESC_PATH",
    r"C:\Users\ARKO\Downloads\Field Descriptions.xlsx"
))

# ── Feature Store Paths ───────────────────────────────────────────────────────
CLEANED_DATA_PATH = FEATURE_STORE_DIR / "cleaned_data.parquet"
USER_OVERVIEW_PATH = FEATURE_STORE_DIR / "user_overview.parquet"
USER_ENGAGEMENT_PATH = FEATURE_STORE_DIR / "user_engagement.parquet"
USER_EXPERIENCE_PATH = FEATURE_STORE_DIR / "user_experience.parquet"
USER_SATISFACTION_PATH = FEATURE_STORE_DIR / "user_satisfaction.parquet"

# ── MySQL Configuration ───────────────────────────────────────────────────────
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "database": os.getenv("MYSQL_DATABASE", "telecom_analytics"),
    "user": os.getenv("MYSQL_USER", "telecom_user"),
    "password": os.getenv("MYSQL_PASSWORD", "Password123#"),
}

# ── MLflow Configuration ──────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "telecom_satisfaction")

# ── Model Parameters ──────────────────────────────────────────────────────────
ENGAGEMENT_K = 3          # k-means clusters for engagement
EXPERIENCE_K = 3          # k-means clusters for experience
SATISFACTION_K = 2        # k-means clusters for satisfaction
ELBOW_MAX_K = 10          # max k to test for elbow method
RANDOM_STATE = 42

# ── Column Groups ─────────────────────────────────────────────────────────────
USER_ID_COL = "MSISDN/Number"

DURATION_COL = "Dur. (ms)"

APP_COLS = {
    "Social Media": ["Social Media DL (Bytes)", "Social Media UL (Bytes)"],
    "Google":       ["Google DL (Bytes)",       "Google UL (Bytes)"],
    "Email":        ["Email DL (Bytes)",         "Email UL (Bytes)"],
    "YouTube":      ["Youtube DL (Bytes)",       "Youtube UL (Bytes)"],
    "Netflix":      ["Netflix DL (Bytes)",       "Netflix UL (Bytes)"],
    "Gaming":       ["Gaming DL (Bytes)",        "Gaming UL (Bytes)"],
    "Other":        ["Other DL (Bytes)",         "Other UL (Bytes)"],
}

EXPERIENCE_COLS = [
    "TCP DL Retrans. Vol (Bytes)",
    "TCP UL Retrans. Vol (Bytes)",
    "Avg RTT DL (ms)",
    "Avg RTT UL (ms)",
    "Avg Bearer TP DL (kbps)",
    "Avg Bearer TP UL (kbps)",
    "Handset Type",
]

THROUGHPUT_COLS = ["Avg Bearer TP DL (kbps)", "Avg Bearer TP UL (kbps)"]
RTT_COLS = ["Avg RTT DL (ms)", "Avg RTT UL (ms)"]
TCP_COLS = ["TCP DL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes)"]

TOTAL_DL_COL = "Total DL (Bytes)"
TOTAL_UL_COL = "Total UL (Bytes)"
HANDSET_TYPE_COL = "Handset Type"
HANDSET_MFR_COL = "Handset Manufacturer"
