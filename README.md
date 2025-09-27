# AliRec 项目总览

## 项目背景
- 2014 年阿里巴巴移动电商业务快速发展，移动端行为数据丰富。
- 本项目基于脱敏后的移动电商平台用户行为数据构建个性化推荐模型。

## 核心任务
- 使用训练期用户行为数据构建推荐模型，推断用户在预测日的购买行为。
- 输出用户-商品对 (user_id, item_id)，格式为制表符分隔。
- 以 F1 值作为唯一评测指标，综合考虑精确率与召回率。

## 数据简介
- 行为数据 (`tianchi_mobile_recommend_train_user`)：包含用户浏览、收藏、加购、购买等行为，字段含 user_id、item_id、behavior_type、user_geohash、item_category、time。
- 商品子集 (`tianchi_mobile_recommend_train_item`)：包含子集中商品的 item_id、item_geohash、item_category。
- 训练期：2014-11-18 至 2014-12-18；预测期：2014-12-19。

## 目录结构
- `data/`: 原始数据文件；体积较大，分为行为数据 A/B 两部分及商品子集文件。
- `processed/`: 数据清洗输出目录，包含 CSV、Parquet 以及清洗摘要。
  - `behavior_cleaned.csv` / `item_cleaned.csv`
  - `behavior_parquet/`（分块 Parquet，命名格式 `part_<源文件>_<序号>.parquet`）
  - `item_parquet/items.parquet`
  - `cleaning_summary.json`
- `processed/eda/`: DuckDB EDA 结果（`eda_summary.json`、`daily_behavior_counts.csv`、`behavior_type_breakdown.csv`）。
- `processed/recall/`: 候选召回结果存放目录（如 `candidates_*.parquet`）。
- `processed/features/`: 特征表输出目录（如 `sample_features.parquet`）。
- `src/evaluation/`: 离线评估工具，提供指标函数 (`metrics.py`)、文件加载 (`io.py`) 与命令行接口 (`cli.py`)。
- `src/recall/`: 候选召回模块，包含数据加载 (`loader.py`)、策略基类 (`base.py`)、召回策略实现（`strategies/`）及 CLI 入口 (`cli.py`)。
- `src/features/`: 特征工程模块，包含数据上下文 (`loader.py`)、特征生成器 (`transforms/`)、特征构建器 (`builder.py`) 与 CLI (`cli.py`)。
- `src/models/`: 模型训练模块，现包含 LightGBM 基线实现 (`baseline.py`, `train_lightgbm.py`)。
- `src/workflows/`: 封装召回+特征+标签的自动化流程脚本。
- `src/`: 项目源码，按职责拆分模块。
  - `src/data_cleaning.py`：数据清洗入口脚本，支持配置化运行。
  - `src/configuration/`：集中管理配置解析。
  - `src/io_utils.py`：通用 IO 函数，封装目录创建、Parquet 写出等。

## 数据处理流程
- **运行清洗脚本**
  - 默认执行：`python -m src.data_cleaning`
  - 指定配置：`python -m src.data_cleaning --config configs/data_cleaning.yaml`
- **生成 EDA**：`python -m src.eda_overview` 输出 `processed/eda/` 下的统计摘要与 CSV。
- **候选召回**：`python -m src.recall.cli 2014-12-19 --max-per-user 300` 可生成面向预测日的候选集，参数支持自定义时间窗口、用户子集、输出路径。
- **特征构建**：`python -m src.features.cli 2014-12-19 processed/recall/candidates_sample.parquet --output processed/features/sample_features.parquet --label-day 2014-12-19` 将召回结果拼接用户/商品统计特征，并通过 `--label-day` 参数基于预测日购买记录写入 `label` 列。
- **训练/验证集生成**：`python -m src.workflows.build_dataset train 2014-12-18 --label-day 2014-12-19 --output-dir processed/datasets` 会自动运行召回、特征、标签步骤，输出特征表到 `processed/datasets/train/`（`val/test` 模式同理）。
- **基线训练**：`python -m src.models.train_lightgbm processed/datasets/train/features_2014-12-18.parquet --valid processed/datasets/val/features_2014-12-19.parquet --output-model processed/models/lightgbm.txt --prediction-output processed/predictions/lightgbm_val.parquet` 训练 LightGBM 模型并输出验证集预测（当前召回覆盖不足，指标为 0）。