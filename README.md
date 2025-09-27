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
- `src/evaluation/`: 离线评估工具，提供指标函数 (`metrics.py`)、文件加载 (`io.py`) 与命令行接口 (`cli.py`)。
- `src/evaluation/`: 离线指标评估工具，包含指标函数 (`metrics.py`)、输入输出 (`io.py`) 与 CLI 入口 (`cli.py`)。
- `src/`: 项目源码，按职责拆分模块。
  - `src/data_cleaning.py`：数据清洗入口脚本，支持配置化运行。
  - `src/configuration/`：集中管理配置解析。
  - `src/io_utils.py`：通用 IO 函数，封装目录创建、Parquet 写出等。
- `configs/`: YAML 配置文件，当前包含 `data_cleaning.yaml`。
- `notebooks/`: 规划用于探索分析（EDA）的笔记本。
- `README.md`: 当前文档。
- `Task.md`: 官方任务说明。

## 数据处理流程
2. **运行清洗脚本**
   - 默认执行：`python -m src.data_cleaning`
   - 指定配置：`python -m src.data_cleaning --config configs/data_cleaning.yaml`
3. **输出内容**
   - CSV：清洗后的行为/商品明细。
   - Parquet：按分块输出的列式数据，可直接供 pandas/polars/DuckDB/Spark 使用。
   - 摘要：`processed/cleaning_summary.json` 记录原始/清洗行数、异常计数，便于审计。
4. **路径解析**
   - 配置文件支持相对/绝对路径，最终由 `DataCleaningConfig` 统一解析到项目根目录下，方便本地与集群环境保持一致。


## 使用说明
- 依赖环境：conda activate AI 
- 数据读取：优先使用分块读取或列式存储（Parquet），避免内存瓶颈。
- **数据格式策略**：统一以 Parquet 作为中间层；若需要缓存模型输入，再根据场景导出 `npz`/`pt` 等格式，保证灵活性与兼容性。

## 后续更新
- 随进度补充详细的模块文档（数据处理、建模、测试）。
- 在 README 中记录关键决策、遇到的问题与解决方案。