"""
Export user_scores to MySQL – standalone script.
Run from the project root:
    python scripts/export_scores_to_mysql.py
"""

# ── Cell 1: Imports & connection ─────────────────────────────────────────────
import sys
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).parents[1]))
import config

# quote_plus is required: the '#' in Password123# is a URL fragment delimiter
# and silently truncates the connection string without it.
password = quote_plus(config.MYSQL_CONFIG["password"])
host     = config.MYSQL_CONFIG["host"]      # 127.0.0.1  (from .env)
port     = config.MYSQL_CONFIG["port"]      # 3307       (from .env)
user     = config.MYSQL_CONFIG["user"]
database = config.MYSQL_CONFIG["database"]

engine = create_engine(
    f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}",
    echo=False,
)
print(f"Connecting to  mysql://{user}@{host}:{port}/{database} ...")

# ── Cell 2: Build the scores DataFrame ───────────────────────────────────────
from src.data.loader import DataLoader
from src.data.feature_engineering import FeatureEngineer
from src.analysis.engagement import EngagementAnalysis
from src.analysis.experience import ExperienceAnalysis
from src.analysis.satisfaction import SatisfactionAnalysis

loader  = DataLoader()
df_raw  = loader.load_or_create_cleaned()
fe      = FeatureEngineer(df_raw)

user_eng = fe.user_engagement_features()
user_app = fe.app_traffic_features()
user_exp = fe.user_experience_features()

eng = EngagementAnalysis(user_eng, user_app)
eng.run_kmeans(k=config.ENGAGEMENT_K)

exp = ExperienceAnalysis(user_exp)
exp.run_kmeans(k=config.EXPERIENCE_K)

sat   = SatisfactionAnalysis(eng.eng, exp.exp)
table = sat.build_satisfaction_table()

# Rename MSISDN/Number → customer_id to match assignment schema
export_df = table[
    [config.USER_ID_COL, "engagement_score", "experience_score", "satisfaction_score"]
].rename(columns={config.USER_ID_COL: "customer_id"})

print(f"\nRows to export : {len(export_df):,}")
print(export_df.head())

# ── Cell 3: Push to MySQL ─────────────────────────────────────────────────────
export_df.to_sql(name="user_scores", con=engine, if_exists="replace", index=False)

# ── Cell 4: Verify ────────────────────────────────────────────────────────────
with engine.connect() as conn:
    result = pd.read_sql(
        text("SELECT * FROM user_scores LIMIT 5"),
        conn,
    )

print("\n✅ Export Successful")
print("\nFirst 5 rows read back from MySQL 'user_scores':")
print(result.to_string(index=False))
