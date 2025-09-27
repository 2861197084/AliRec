"""基于 DuckDB 的快速 EDA 汇总脚本。

从清洗后的 Parquet 数据中提取核心统计量，生成 JSON 与 CSV
摘要，供后续建模与报告使用。默认读取 `processed/behavior_parquet`
与 `processed/item_parquet/items.parquet`。

运行示例：
```
python -m src.eda_overview
```
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed"
BEHAVIOR_PARQUET_GLOB = (PROCESSED_DIR / "behavior_parquet" / "*.parquet").as_posix()
ITEM_PARQUET_PATH = (PROCESSED_DIR / "item_parquet" / "items.parquet").as_posix()
OUTPUT_DIR = PROCESSED_DIR / "eda"
SUMMARY_JSON_PATH = OUTPUT_DIR / "eda_summary.json"
DAILY_CSV_PATH = OUTPUT_DIR / "daily_behavior_counts.csv"
BEHAVIOR_TYPE_CSV_PATH = OUTPUT_DIR / "behavior_type_breakdown.csv"


def _rows_to_dicts(rows: List[tuple], columns: List[str]) -> List[Dict[str, Any]]:
    return [dict(zip(columns, row)) for row in rows]


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_eda() -> None:
    _ensure_output_dir()
    con = duckdb.connect(database=":memory:")

    behavior_rel = f"read_parquet('{BEHAVIOR_PARQUET_GLOB}')"
    item_rel = f"read_parquet('{ITEM_PARQUET_PATH}')"

    total_rows = con.execute(f"SELECT COUNT(*) FROM {behavior_rel}").fetchone()[0]
    distinct_users = con.execute(f"SELECT COUNT(DISTINCT user_id) FROM {behavior_rel}").fetchone()[0]
    distinct_items = con.execute(f"SELECT COUNT(DISTINCT item_id) FROM {behavior_rel}").fetchone()[0]
    distinct_pairs = con.execute(
        f"SELECT COUNT(*) FROM (SELECT user_id, item_id FROM {behavior_rel} GROUP BY user_id, item_id)"
    ).fetchone()[0]
    purchase_pairs = con.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT user_id, item_id
            FROM {behavior_rel}
            WHERE behavior_type = 4
            GROUP BY user_id, item_id
        )
        """
    ).fetchone()[0]
    purchasing_users = con.execute(
        f"SELECT COUNT(DISTINCT user_id) FROM {behavior_rel} WHERE behavior_type = 4"
    ).fetchone()[0]
    purchasing_items = con.execute(
        f"SELECT COUNT(DISTINCT item_id) FROM {behavior_rel} WHERE behavior_type = 4"
    ).fetchone()[0]

    behavior_counts_rows = con.execute(
        f"SELECT behavior_type, COUNT(*) AS event_count FROM {behavior_rel} GROUP BY 1 ORDER BY 1"
    ).fetchall()
    behavior_counts = _rows_to_dicts(behavior_counts_rows, ["behavior_type", "event_count"])

    daily_counts_df = con.execute(
        f"""
        SELECT
            CAST(date_trunc('day', time) AS DATE) AS day,
            behavior_type,
            COUNT(*) AS event_count
        FROM {behavior_rel}
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchdf()
    daily_counts_df["day"] = daily_counts_df["day"].astype(str)
    daily_counts_df.to_csv(DAILY_CSV_PATH, index=False, encoding="utf-8")

    behavior_type_df = pd.DataFrame(behavior_counts)
    behavior_type_df.to_csv(BEHAVIOR_TYPE_CSV_PATH, index=False, encoding="utf-8")

    top_purchase_categories_rows = con.execute(
        f"""
        SELECT
            item_category,
            COUNT(*) AS purchase_events
        FROM {behavior_rel}
        WHERE behavior_type = 4
        GROUP BY 1
        ORDER BY purchase_events DESC
        LIMIT 10
        """
    ).fetchall()
    top_purchase_categories = _rows_to_dicts(
        top_purchase_categories_rows, ["item_category", "purchase_events"]
    )

    hourly_activity_rows = con.execute(
        f"""
        SELECT
            CAST(date_trunc('hour', time) AS TIMESTAMP) AS hour,
            COUNT(*) AS event_count
        FROM {behavior_rel}
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchall()
    hourly_activity = [
        {"hour": row[0].strftime("%Y-%m-%d %H:%M:%S"), "event_count": row[1]}
        for row in hourly_activity_rows
    ]

    item_total = con.execute(f"SELECT COUNT(*) FROM {item_rel}").fetchone()[0]
    item_distinct_categories = con.execute(
        f"SELECT COUNT(DISTINCT item_category) FROM {item_rel}"
    ).fetchone()[0]
    item_missing_geohash = con.execute(
        f"SELECT COUNT(*) FROM {item_rel} WHERE item_geohash IS NULL OR item_geohash = ''"
    ).fetchone()[0]

    top_item_categories_rows = con.execute(
        f"""
        SELECT item_category, COUNT(*) AS item_count
        FROM {item_rel}
        GROUP BY 1
        ORDER BY item_count DESC
        LIMIT 10
        """
    ).fetchall()
    top_item_categories = _rows_to_dicts(top_item_categories_rows, ["item_category", "item_count"])

    avg_events_per_user = float(total_rows) / distinct_users if distinct_users else 0.0
    avg_events_per_item = float(total_rows) / distinct_items if distinct_items else 0.0
    conversion_rate = (
        float(purchase_pairs) / distinct_pairs if distinct_pairs else 0.0
    )

    summary: Dict[str, Any] = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "behavior_overview": {
            "total_rows": int(total_rows),
            "distinct_users": int(distinct_users),
            "distinct_items": int(distinct_items),
            "distinct_user_item_pairs": int(distinct_pairs),
            "purchase_user_item_pairs": int(purchase_pairs),
            "purchasing_users": int(purchasing_users),
            "purchasing_items": int(purchasing_items),
            "avg_events_per_user": avg_events_per_user,
            "avg_events_per_item": avg_events_per_item,
            "pair_level_conversion_rate": conversion_rate,
            "behavior_counts": behavior_counts,
            "top_purchase_categories": top_purchase_categories,
            "hourly_activity": hourly_activity,
        },
        "item_overview": {
            "total_items": int(item_total),
            "distinct_categories": int(item_distinct_categories),
            "missing_geohash_items": int(item_missing_geohash),
            "top_item_categories": top_item_categories,
        },
        "artifacts": {
            "daily_behavior_counts_csv": DAILY_CSV_PATH.as_posix(),
            "behavior_type_breakdown_csv": BEHAVIOR_TYPE_CSV_PATH.as_posix(),
        },
    }

    with SUMMARY_JSON_PATH.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(f"EDA 摘要已生成：{SUMMARY_JSON_PATH}")
    print(f"按日行为统计已输出：{DAILY_CSV_PATH}")
    print(f"行为类型分布已输出：{BEHAVIOR_TYPE_CSV_PATH}")


def main() -> None:
    run_eda()


if __name__ == "__main__":
    main()


