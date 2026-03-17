# CLAUDE.md — AIQ (NVIDIA AIRA) / RCA Agent

## 项目概述

基于 NVIDIA NeMo Agent Toolkit (nat) + LangGraph 构建的多阶段深度研究系统。原始设计为 3 阶段 Web 研究 + 报告生成流水线，现已适配 RCA（Root Cause Analysis）评测场景。

**RCA 评测模式**：保留 AIRA 原始多阶段流水线架构（generate_queries → data_research → summarize → reflect → finalize），仅将 RAG/Tavily 搜索替换为 DuckDB parquet 工具，通过 `agent_runner.py` 接入 RolloutRunner 统一评测管道。

---

## 两种运行模式

### 1. 原始模式：Deep Research 服务（nat serve）

3 阶段流水线，通过 REST API 提供服务：

```
Stage 1: generate_query    → 生成研究查询列表
Stage 2: generate_summary  → Web 研究 + 报告撰写 + 反思 + 终稿
Stage 3: artifact_qa       → 基于报告的 Q&A
```

入口：`uv run nat serve --config_file configs/config.yml --host 0.0.0.0 --port 3838`

### 2. RCA 评测模式（agent_runner.py）

保留 AIRA 多阶段流水线，替换搜索工具为 parquet 工具，与 RolloutRunner 标准接口对接：

```
START → generate_queries → data_research → summarize_sources → reflect_on_summary → finalize_summary → END
```

入口：`echo '{"question":...}' | uv run --no-project --python 3.12 --with "..." python agent_runner.py`

---

## RCA 评测图结构（保留 AIRA 多阶段架构）

```
START
  └─► generate_queries        # 对应 AIRA Stage 1: generate_query
        │  事件描述 → LLM → 1 个调查子查询（JSON 列表）
        │
        └─► data_research      # 替代 AIRA Stage 2: web_research
              │  每个查询 → LLM + parquet 工具（tool-calling 循环）→ 数据发现
              │  工具：think_tool, list_tables_in_directory, get_schema, query_parquet_files
              │  结果格式化为 XML <sources>（对应 AIRA deduplicate_and_format_sources）
              │
              └─► summarize_sources    # 对应 AIRA summarize_sources
                    │  数据发现 → LLM → RCA 分析报告（running_summary）
                    │
                    └─► reflect_on_summary   # 对应 AIRA reflect_on_summary
                          │  分析报告 → LLM → 发现知识缺口 → 补充数据探索
                          │  循环 num_reflections 次（默认 1 次）
                          │
                          └─► finalize_summary   # 对应 AIRA finalize_summary
                                │  running_summary → compress prompts → CausalGraph JSON
                                └─► END
```

### AIRA 节点映射

| AIRA 原始节点 | RCA 适配节点 | 关键变化 |
|--------------|-------------|---------|
| `generate_query` (Stage 1) | `generate_queries` | prompt 改为 RCA 调查查询 |
| `web_research` (Stage 2) | `data_research` | RAG/Tavily → parquet tool-calling 循环 |
| `summarize_sources` | `summarize_sources` | prompt 改为 RCA 分析 |
| `reflect_on_summary` | `reflect_on_summary` | follow-up 用 parquet 工具替代 web 搜索 |
| `finalize_summary` | `finalize_summary` | 输出 CausalGraph JSON 替代 markdown 报告 |

### data_research 内部工具调用循环

每个查询独立运行一个 tool-calling mini-agent（替代 AIRA 的 `process_single_query`）：

```
SystemMessage(system_prompt + "---" + DATA_RESEARCH_SP) + HumanMessage(query + data_dir)
  └─► LLM.invoke (bind_tools) → AIMessage
        ├─► 有 tool_calls → 执行工具 → ToolMessage → 回到 LLM（最多 15 轮）
        └─► 无 tool_calls → 返回 findings
```

> `system_prompt`（RCA_ANALYSIS_SP）与 `DATA_RESEARCH_SP` 合并注入，与 thinkdepthai 的
> `combined_sp = system_prompt + "---" + rca_think_prompt` 一致。
> `question`（augmented_question + data_dir）注入 `generate_queries` 和 `reflect_on_summary`。

---

## Tools 定义

### RCA 工具集（agent_runner.py 中使用）

| Tool | 来源 | 功能 |
|------|------|------|
| `think_tool` | agent_runner.py 内联 | 反思占位工具（返回记录的思考内容） |
| `list_tables_in_directory` | rca_tools.py | 列出目录中所有 parquet 文件及元数据 |
| `get_schema` | rca_tools.py | 获取单个/多个 parquet 文件的 schema |
| `query_parquet_files` | rca_tools.py | 用 DuckDB SQL 查询 parquet 数据，token 限制 5000 |

> 不使用 tavily_search，与 RCA_ANALYSIS_SP 描述一致。

---

## Prompt 体系（适配自 AIRA prompts.py）

| agent_runner.py Prompt | 对应 AIRA prompt | 用途 |
|------------------------|------------------|------|
| `RCA_QUERY_WRITER` | `query_writer_instructions` | 生成调查子查询 |
| `DATA_RESEARCH_SP` | （新增，替代 RAG/Tavily） | 数据探索系统提示 |
| `RCA_SUMMARIZER` | `summarizer_instructions` | 首次汇总数据发现 |
| `RCA_REPORT_EXTENDER` | `report_extender` | 扩展已有分析 |
| `RCA_REFLECTION` | `reflection_instructions` | 发现知识缺口 |
| `compress_system_prompt` | `finalize_report` | 生成最终 CausalGraph JSON |

> stdin 传入的 `compress_system_prompt` / `compress_user_prompt` 用于 `finalize_summary` 节点。
> `question`（augmented + data_dir 增强）传入 `generate_queries` 和 `reflect_on_summary` 节点。
> `system_prompt`（RCA_ANALYSIS_SP）注入 `run_data_exploration()` 的 SystemMessage。

---

## 环境管理

**工具：`uv`**（见 `pyproject.toml`）

```bash
# 安装依赖
uv sync

# 额外安装 RCA 评测所需的 duckdb
uv pip install duckdb

# 运行 agent_runner
echo '...' | uv run python agent_runner.py
```

Python 要求：`>=3.12`

主要依赖：`langgraph`, `langchain-openai`, `langchain-core`, `duckdb`, `python-dotenv`

---

## 环境变量

在项目根目录创建 `.env` 文件：

```
OPENAI_API_KEY=sk-...           # 必填，驱动 LLM 调用
OPENAI_BASE_URL=https://...     # 可选，自定义 API 地址（如 kimi）
```

---

## RCA 评测 stdin/stdout 接口

**stdin (6 字段):**
```json
{
  "question": "augmented_question 原文",
  "system_prompt": "RCA_ANALYSIS_SP (已 format date)",
  "user_prompt": "RCA_ANALYSIS_UP (已 format incident_description)",
  "compress_system_prompt": "COMPRESS_FINDINGS_SP",
  "compress_user_prompt": "COMPRESS_FINDINGS_UP",
  "data_dir": "/path/to/eval-data/<exp_id>/data_XXXXXXXX"
}
```

**stdout (最后一行 JSON):**
```json
{
  "output": "{\"nodes\": [...], \"edges\": [...], \"root_causes\": [...]}",
  "trajectory": [
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "content": "...", "tool_call_id": "..."}
  ]
}
```

**关键要求：**
- `output` 必须是纯 JSON，不能有 markdown 代码块包裹
- trajectory 必须是 OpenAI role 格式（影响 tool_bonus 计算）
- 成功必须 exit 0

---

## 数据流（RCA 评测模式）

```
stdin JSON
  │
  ▼
agent_runner.py::main()
  │  解析 payload → 构建 config["configurable"]
  │
  ▼
build_agent()
  │  构建 AIRA 多阶段 StateGraph:
  │  generate_queries → data_research → summarize_sources
  │    → reflect_on_summary → finalize_summary
  │
  ▼
agent.invoke(initial_state, config)
  │  Stage 1: 事件描述 → 1 个调查子查询
  │  Stage 2: 每个查询 → parquet 工具调用循环 → 数据发现
  │           数据发现 → RCA 分析摘要
  │           分析摘要 → 反思 → 补充调查 → 更新摘要
  │           最终摘要 → compress prompts → CausalGraph JSON
  │
  ▼
stdout: {"output": strip_markdown_json(final_report), "trajectory": [...]}
```

---

## 原始项目结构

```
aiq/
├── agent_runner.py              # RCA 评测接口（新增）
├── rca_tools.py                 # DuckDB parquet 工具（新增，来自 Deep_Research）
├── CLAUDE.md                    # 本文档
├── pyproject.toml               # 依赖声明（uv 管理）
├── .env                         # API 密钥
│
├── aira/src/aiq_aira/           # 原始 AIRA 包
│   ├── functions/
│   │   ├── generate_queries.py  # Stage 1: 查询生成图
│   │   ├── generate_summary.py  # Stage 2: 研究 + 报告图
│   │   └── artifact_qa.py       # Stage 3: Q&A 图
│   ├── nodes.py                 # LangGraph 节点实现
│   ├── schema.py                # Pydantic 模型 + AIRAState
│   ├── prompts.py               # 所有 prompt 模板
│   ├── tools.py                 # RAG + Tavily 搜索工具
│   ├── search_utils.py          # 搜索辅助函数
│   ├── report_gen_utils.py      # 报告生成辅助
│   ├── register.py              # nat 插件注册
│   └── constants.py             # 常量
│
├── configs/
│   ├── config.yml               # nat 服务配置
│   └── security_config.yml      # 安全模式
│
└── data/                        # 样例数据
```

---

## 关键约束

| 约束 | 值 |
|------|---|
| LangGraph 递归限制 | 100 |
| DuckDB 结果 token 限制 | 5000 tokens |
| Python 版本 | >=3.12 |
| 模型（RCA 评测） | openai/claude-sonnet-4-5-20250929 |

---

## 常用命令

```bash
# === RCA 评测 ===
# 冒烟测试
cd /home/nn/SOTA-agents/RolloutRunner
python scripts/run_rollout.py --agent aiq --source_exp_id <exp_id> --limit 1

# 全量运行
nohup python -u scripts/run_rollout.py --agent aiq --source_exp_id <exp_id> \
  > rollout_aiq.log 2>&1 &

# === 原始服务模式 ===
cd /home/nn/SOTA-agents/aiq
uv run nat serve --config_file configs/config.yml --host 0.0.0.0 --port 3838
```
