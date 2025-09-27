"""召回结果生成命令行入口。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.recall.loader import RecallDataContext, create_default_context
from src.recall.pipeline import RecallPipeline, RecallPipelineConfig
from src.recall.strategies.recent_interaction import RecentInteractionConfig, RecentInteractionStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成用户候选商品集合")
    parser.add_argument("cutoff", type=str, help="预测日（YYYY-MM-DD），仅使用此前数据")
    parser.add_argument("--users", type=Path, default=None, help="可选，指定需要召回的用户列表（单列 user_id TSV）")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed/recall/candidates.parquet"),
        help="输出 Parquet 路径",
    )
    parser.add_argument("--user-limit", type=int, default=None, help="可选，限制参与召回的用户数量")
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--max-per-user", type=int, default=300)
    return parser.parse_args()


def load_users(user_file: Path | None, context: RecallDataContext, user_limit: int | None) -> list[int]:
    if user_file is None:
        con = context.connect()
        limit_clause = "" if user_limit is None else f" LIMIT {user_limit}"
        query = (
            "SELECT DISTINCT user_id FROM read_parquet('{pattern}') ORDER BY user_id{limit}".format(
                pattern=context.behavior_pattern, limit=limit_clause
            )
        )
        rows = con.execute(query).fetchall()
        return [int(row[0]) for row in rows]

    df = pd.read_csv(user_file, header=None, names=["user_id"], dtype="int64")
    if user_limit is not None:
        return df["user_id"].head(user_limit).tolist()
    return df["user_id"].tolist()


def main() -> None:
    args = parse_args()
    cutoff = datetime.fromisoformat(args.cutoff)
    context = create_default_context()
    user_ids = load_users(args.users, context, args.user_limit)

    strategy = RecentInteractionStrategy(
        context=context,
        cutoff=cutoff,
        config=RecentInteractionConfig(lookback_days=args.lookback_days, max_candidates_per_user=args.max_per_user),
    )
    pipeline = RecallPipeline([strategy], config=RecallPipelineConfig(max_per_user=args.max_per_user))
    result_df = pipeline.run(user_ids)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path, index=False)
    print(f"召回结果已生成：{output_path}，共 {len(result_df)} 条候选")


if __name__ == "__main__":
    main()

