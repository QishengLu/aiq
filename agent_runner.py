#!/usr/bin/env python
"""
agent_runner.py — AIQ (NVIDIA AIRA) RCA 测评接口

保留 AIRA 原始多阶段 LangGraph 流水线架构：
  Stage 1: generate_queries   → 将事件描述分解为调查子查询（对应 AIRA generate_query）
  Stage 2: data_research      → 用 parquet 工具探索数据（替代 AIRA web_research）
           summarize_sources   → 汇总数据发现（对应 AIRA summarize_sources）
           reflect_on_summary  → 发现知识缺口并补充调查（对应 AIRA reflect_on_summary）
           finalize_summary    → 生成最终 CausalGraph JSON（对应 AIRA finalize_summary）

工具替换：RAG/Tavily Web Search → DuckDB Parquet 工具（与 thinkdepthai 相同）
模型：kimi-k2-0905-preview
接口：RolloutRunner stdin/stdout 标准接口

stdin:  JSON { question, system_prompt, user_prompt,
               compress_system_prompt, compress_user_prompt, data_dir }
stdout: JSON { output (CausalGraph JSON), trajectory (OpenAI 格式) }
"""
import json
import logging
import operator
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Annotated

sys.path.insert(0, "/home/nn/SOTA-agents/RolloutRunner")
from src.usage_tracker import UsageTracker

_tracker = UsageTracker()
_tracker.install_openai_hooks()

# 根据模型选择 hook：Claude 走 Anthropic SDK，其余走 OpenAI SDK
_RCA_MODEL = os.environ.get("RCA_MODEL", "claude-sonnet-4-6")
if _RCA_MODEL.startswith("claude"):
    _tracker.install_anthropic_hooks()

# 清理 RolloutRunner 路径和 src 模块缓存，避免与本项目的 src 包冲突
sys.path.remove("/home/nn/SOTA-agents/RolloutRunner")
for _mod in list(sys.modules):
    if _mod == "src" or _mod.startswith("src."):
        del sys.modules[_mod]


from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from model_factory import create_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig

from rca_tools import get_schema, list_tables_in_directory, query_parquet_files

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

# ── 模型配置 ──────────────────────────────────────────────────────────────

# MODEL_NAME = "openai:doubao-seed-2-0-pro-260215"
# MODEL_NAME = "openai:kimi-k2-0905-preview"
# MODEL_NAME = "openai:openai/claude-sonnet-4-6"
# MODEL_NAME = "openai:gemini-3.1-pro-preview"
# MODEL_NAME = "openai:openai/claude-sonnet-4-6"
# MODEL_NAME = "openai:claude-sonnet-4-6"

RCA_MODEL = os.environ.get("RCA_MODEL", "claude-sonnet-4-6")


def _make_model(max_tokens: int = 32768):
    """Create LLM via model_factory. Model name from RCA_MODEL env var."""
    return create_model(RCA_MODEL, max_tokens=max_tokens)


# ── think_tool（与 thinkdepthai 相同）──────────────────────────────────────

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each round of data queries to analyze results and plan next steps.

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded
    """
    return f"Reflection recorded: {reflection}"


# ── RCA 工具集 ────────────────────────────────────────────────────────────

RCA_TOOLS = [think_tool, list_tables_in_directory, get_schema, query_parquet_files]
RCA_TOOLS_BY_NAME = {t.name: t for t in RCA_TOOLS}


# ── State（保留 AIRA AIRAState 结构）──────────────────────────────────────

class RCAState(TypedDict):
    """
    对应 AIRA 的 AIRAState，适配 RCA 场景。

    AIRA 原始字段映射：
      queries               → queries（调查查询列表）
      web_research_results  → data_research_results（数据探索结果）
      citations             → （不需要，RCA 无引用）
      running_summary       → running_summary（运行中的 RCA 分析）
      final_report          → final_report（最终 CausalGraph JSON）

    新增字段：
      all_tool_messages     → 工具调用轨迹（用于 trajectory 输出）
    """
    queries: list[dict]
    data_research_results: list[str]
    running_summary: str
    final_report: str
    all_tool_messages: Annotated[list, operator.add]  # 跨节点累积


# ── Prompts（适配自 AIRA prompts.py，改为 RCA 场景）──────────────────────

# 对应 AIRA query_writer_instructions
RCA_QUERY_WRITER = """You are the investigation-query architect for an RCA (Root Cause Analysis) agent that analyzes microservice incidents using telemetry data (logs, metrics, traces in parquet format).

Given an incident description, generate {number_of_queries} investigation queries to systematically explore the telemetry data.

# Incident Description
{incident}

# Instructions
- Design queries that cover different investigation angles:
  * Service error rates and HTTP status codes
  * Latency anomalies and response time spikes
  * Log error patterns and exception messages
  * Trace call chains and service dependencies
  * Resource utilization (CPU, memory) stress indicators
- Each query should be specific enough to guide targeted SQL queries on parquet data
- Format your response as a JSON list:

```json
[
    {{"query": "Investigate error rates across services by comparing normal vs abnormal metrics", "report_section": "Error Analysis", "rationale": "Identify which services have elevated error rates during the incident"}},
    {{"query": "Analyze trace data to find latency spikes and failing call chains", "report_section": "Trace Analysis", "rationale": "Trace the propagation path of the failure"}}
]
```"""

# 对应 AIRA 的 DATA_RESEARCH 系统提示（替代 web_research 中的 RAG/Tavily）
DATA_RESEARCH_SP = """You are an RCA data analyst investigating a microservice incident.
Your task is to explore telemetry data (parquet files) to find evidence related to a specific investigation query.

You have these tools:
- list_tables_in_directory: List available parquet files with row counts
- get_schema: Get column names and types of parquet files
- query_parquet_files: Execute SQL queries on parquet data
- think_tool: Record your reflections and analysis

## Investigation Protocol
1. First call list_tables_in_directory to discover available data
2. Use get_schema to understand the structure of relevant files
3. Write targeted SQL queries to find anomalies
4. Call think_tool after every round of queries to analyze findings
5. Compare normal vs abnormal data to identify deviations
6. Focus on: error counts, latency spikes, service names, timestamps

## Important
- The root cause is the UPSTREAM service that INITIATED the failure
- Look for the earliest anomaly in the timeline
- Be thorough — continue investigating until you have enough evidence to identify the root cause
- Summarize your key findings clearly at the end
"""

# 对应 AIRA summarizer_instructions
RCA_SUMMARIZER = """Based on all the data exploration results below, create a comprehensive RCA (Root Cause Analysis) report.

# Data Exploration Results
{sources}

# Instructions
1. Identify anomalous services, error patterns, and timing correlations
2. Trace the failure propagation path from root cause to affected services
3. The root cause is the UPSTREAM service that INITIATED the failure, not downstream victims
4. Structure your analysis clearly:
   - ## Incident Overview
   - ## Anomaly Findings (per service)
   - ## Propagation Path
   - ## Root Cause Identification
5. Include specific evidence: error counts, latency values, timestamps, service names
6. Be comprehensive — include ALL relevant findings from the data exploration
"""

# 对应 AIRA report_extender
RCA_REPORT_EXTENDER = """Based on the current RCA analysis draft and newly discovered evidence, update and extend the analysis.

# Current Analysis
{report}

# New Evidence
{source}

# Instructions
1. Preserve the existing analysis structure
2. Incorporate new evidence to strengthen or revise the root cause hypothesis
3. Update the propagation path if new service dependencies are discovered
4. Add any new anomalous services or patterns found
5. Do NOT remove existing findings unless contradicted by new evidence
"""

# 对应 AIRA reflection_instructions
RCA_REFLECTION = """Review the current RCA analysis and identify knowledge gaps that need further investigation.

# Current RCA Analysis
{report}

# Original Incident Description
{topic}

# Instructions
1. What aspects of the incident haven't been fully investigated?
2. Are there services mentioned in the data that haven't been explored?
3. Is the root cause hypothesis well-supported? What additional evidence would strengthen it?
4. Are there alternative root cause hypotheses to consider?
5. Generate a specific follow-up investigation query.

Format your response as JSON:
```json
{{
    "query": "Specific SQL-oriented investigation query for parquet data",
    "report_section": "Section this addresses",
    "rationale": "What information gap this fills"
}}
```"""


# ── Helper: 工具调用数据探索（替代 AIRA 的 process_single_query）──────────

def run_data_exploration(query: str, data_dir: str, system_prompt: str = "") -> tuple[str, list]:
    """
    替代 AIRA 的 process_single_query（search_utils.py）。
    用 LLM + parquet 工具进行数据探索，类似 AIRA 的 search_rag + search_tavily。

    Args:
        system_prompt: RCA 领域系统提示（来自 RolloutRunner 的 RCA_ANALYSIS_SP），
                       与 DATA_RESEARCH_SP 合并，类似 thinkdepthai 的
                       combined_sp = system_prompt + rca_think_prompt

    Returns:
        (findings_text, tool_messages_list)
    """
    model = _make_model()
    model_with_tools = model.bind_tools(RCA_TOOLS)

    # 合并系统提示：RCA 领域指令在前，工具使用指南在后
    # 对应 thinkdepthai 的 combined_sp = system_prompt + "\n\n---\n\n" + rca_think_prompt
    combined_sp = system_prompt + "\n\n---\n\n" + DATA_RESEARCH_SP if system_prompt else DATA_RESEARCH_SP

    messages = [
        SystemMessage(content=combined_sp),
        HumanMessage(content=(
            f"Investigation query: {query}\n\n"
            f"Data location: `{data_dir}`\n"
            f"Start by calling `list_tables_in_directory(directory=\"{data_dir}\")` "
            f"to discover available parquet files."
        )),
    ]

    all_msgs = []
    MAX_TOOL_ROUNDS = 15
    for _round in range(MAX_TOOL_ROUNDS):
        response = model_with_tools.invoke(messages)
        messages.append(response)
        all_msgs.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            tool_fn = RCA_TOOLS_BY_NAME[tc["name"]]
            result = tool_fn.invoke(tc["args"])
            tool_msg = ToolMessage(
                content=result, name=tc["name"], tool_call_id=tc["id"]
            )
            messages.append(tool_msg)
            all_msgs.append(tool_msg)

    findings = str(response.content) if response.content else ""
    return findings, all_msgs


def deduplicate_and_format_sources(queries: list[dict], findings: list[str]) -> str:
    """
    对应 AIRA 的 deduplicate_and_format_sources（search_utils.py）。
    将每个 query 的探索结果格式化为 XML <sources> 结构。
    """
    root = ET.Element("sources")
    for q, finding in zip(queries, findings):
        source_elem = ET.SubElement(root, "source")
        query_elem = ET.SubElement(source_elem, "query")
        query_elem.text = q["query"] if isinstance(q, dict) else str(q)
        answer_elem = ET.SubElement(source_elem, "answer")
        answer_elem.text = finding
    return ET.tostring(root, encoding="unicode")


def strip_think_tags(text: str) -> str:
    """清理 <think>...</think> 标签（对应 AIRA report_gen_utils.py 的逻辑）。"""
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]
    while "</think>" in text:
        end = text.find("</think>") + len("</think>")
        text = text[end:]
    return text


# ── Node 1: generate_queries（对应 AIRA Stage 1: generate_query）─────────

def generate_queries(state: RCAState, config: RunnableConfig) -> dict:
    """
    对应 AIRA 的 generate_query 节点（nodes.py:generate_query）。
    从事件描述生成调查子查询列表。

    AIRA 原始流程：
      topic + report_organization → query_writer_instructions → LLM → parse JSON → GeneratedQuery[]
    RCA 适配：
      incident_description → RCA_QUERY_WRITER → LLM → parse JSON → query dicts
    """
    logger.info("GENERATE QUERIES")
    llm = _make_model()
    # 使用 augmented question 作为事件描述（含数据路径信息）
    incident = config["configurable"].get("question") or config["configurable"]["user_prompt"]
    number_of_queries = config["configurable"].get("number_of_queries", 1)

    prompt = RCA_QUERY_WRITER.format(
        incident=incident, number_of_queries=number_of_queries
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    text = strip_think_tags(str(response.content))

    # 解析 JSON 查询列表（对应 AIRA 的 parse_json_markdown + GeneratedQuery 验证）
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            queries = json.loads(m.group(0))
        except Exception:
            queries = [
                {
                    "query": incident,
                    "report_section": "Full Analysis",
                    "rationale": "Direct investigation",
                }
            ]
    else:
        queries = [
            {
                "query": incident,
                "report_section": "Full Analysis",
                "rationale": "Direct investigation",
            }
        ]

    # 硬截断：LLM 可能忽略 number_of_queries 参数，生成更多查询
    max_queries = number_of_queries
    if len(queries) > max_queries:
        logger.info(f"Truncating {len(queries)} queries to {max_queries}")
        queries = queries[:max_queries]

    logger.info(f"Generated {len(queries)} investigation queries")
    return {"queries": queries}


# ── Node 2: data_research（替代 AIRA Stage 2: web_research）──────────────

def data_research(state: RCAState, config: RunnableConfig) -> dict:
    """
    对应 AIRA 的 web_research 节点（nodes.py:web_research）。
    对每个查询执行 parquet 数据探索（替代 RAG + Tavily 搜索）。

    AIRA 原始流程：
      queries → [process_single_query(RAG + Tavily + relevancy_check)] x N
              → deduplicate_and_format_sources → XML <sources>
    RCA 替换：
      queries → [run_data_exploration(parquet tools)] x N
              → deduplicate_and_format_sources → XML <sources>
    """
    logger.info("STARTING DATA RESEARCH")
    data_dir = config["configurable"]["data_dir"]
    queries = state.get("queries") or []

    system_prompt = config["configurable"].get("system_prompt", "")

    all_findings = []
    all_msgs = []

    for q in queries:
        query_text = q["query"] if isinstance(q, dict) else str(q)
        logger.info(f"Researching: {query_text[:80]}...")
        findings, msgs = run_data_exploration(query_text, data_dir, system_prompt)
        all_findings.append(findings)
        all_msgs.extend(msgs)

    # 格式化为 XML（对应 AIRA deduplicate_and_format_sources）
    research_xml = deduplicate_and_format_sources(queries, all_findings)
    logger.info("Data research complete")

    return {
        "data_research_results": [research_xml],
        "all_tool_messages": all_msgs,
    }


# ── Node 3: summarize_sources（对应 AIRA summarize_sources）──────────────

def summarize_sources(state: RCAState, config: RunnableConfig) -> dict:
    """
    对应 AIRA 的 summarize_sources 节点（nodes.py:summarize_sources）。
    汇总数据探索结果为 RCA 分析报告。

    AIRA 原始流程：
      web_research_results[-1] + existing_summary
        → summarizer_instructions 或 report_extender
        → updated running_summary
    RCA 替换：
      data_research_results[-1] + existing_summary
        → RCA_SUMMARIZER 或 RCA_REPORT_EXTENDER
        → updated running_summary
    """
    logger.info("SUMMARIZE SOURCES")
    llm = _make_model()

    # 取最新的研究结果（对应 AIRA 的 state.web_research_results[-1]）
    results = state.get("data_research_results") or []
    most_recent = results[-1] if results else ""
    existing_summary = state.get("running_summary") or ""

    if existing_summary:
        # 已有摘要 → 扩展（对应 AIRA report_extender prompt）
        prompt = RCA_REPORT_EXTENDER.format(
            report=existing_summary, source=most_recent
        )
    else:
        # 首次摘要（对应 AIRA summarizer_instructions prompt）
        prompt = RCA_SUMMARIZER.format(sources=most_recent)

    response = llm.invoke([HumanMessage(content=prompt)])
    summary = strip_think_tags(str(response.content))

    logger.info("Summary complete")
    return {"running_summary": summary}


# ── Node 4: reflect_on_summary（对应 AIRA reflect_on_summary）────────────

def reflect_on_summary(state: RCAState, config: RunnableConfig) -> dict:
    """
    对应 AIRA 的 reflect_on_summary 节点（nodes.py:reflect_on_summary）。
    发现知识缺口，生成补充查询，执行额外数据探索，更新分析。

    AIRA 原始流程（循环 num_reflections 次）：
      1. reflection_instructions → parse JSON → follow-up GeneratedQuery
      2. process_single_query(RAG + Tavily) → new sources
      3. deduplicate_and_format_sources → append to web_research_results
      4. summarize_report (report_extender) → updated running_summary
    RCA 替换：
      1. RCA_REFLECTION → parse JSON → follow-up query
      2. run_data_exploration(parquet tools) → new findings
      3. deduplicate_and_format_sources → append to data_research_results
      4. RCA_REPORT_EXTENDER → updated running_summary
    """
    logger.info("REFLECTING")
    llm = _make_model()
    data_dir = config["configurable"]["data_dir"]
    num_reflections = config["configurable"].get("num_reflections", 1)
    # 使用 augmented question 作为 topic（与 generate_queries 一致）
    topic = config["configurable"].get("question") or config["configurable"]["user_prompt"]
    system_prompt = config["configurable"].get("system_prompt", "")

    running_summary = state.get("running_summary") or ""
    data_results = list(state.get("data_research_results") or [])
    all_msgs = []

    for i in range(num_reflections):
        logger.info(f"Reflection {i + 1}/{num_reflections}")

        # Step 1: 反思当前分析（对应 AIRA 的 reflection_instructions 调用）
        prompt = RCA_REFLECTION.format(report=running_summary, topic=topic)
        response = llm.invoke([HumanMessage(content=prompt)])
        result_text = strip_think_tags(str(response.content))

        # 解析 follow-up query（对应 AIRA 的 parse_json_markdown）
        m = re.search(r"\{.*\}", result_text, re.DOTALL)
        if m:
            try:
                reflection_obj = json.loads(m.group(0))
                follow_up_query = reflection_obj.get("query", "")
            except Exception:
                follow_up_query = result_text
        else:
            follow_up_query = result_text

        if not follow_up_query:
            break

        logger.info(f"Follow-up query: {follow_up_query[:80]}...")

        # Step 2: 补充数据探索（对应 AIRA reflect 中的 process_single_query）
        findings, msgs = run_data_exploration(follow_up_query, data_dir, system_prompt)
        all_msgs.extend(msgs)

        # Step 3: 格式化并追加（对应 AIRA 的 deduplicate_and_format_sources + append）
        new_source = deduplicate_and_format_sources(
            [{"query": follow_up_query}], [findings]
        )
        data_results.append(new_source)

        # Step 4: 扩展分析（对应 AIRA 的 summarize_report with report_extender）
        extend_prompt = RCA_REPORT_EXTENDER.format(
            report=running_summary, source=new_source
        )
        ext_response = llm.invoke([HumanMessage(content=extend_prompt)])
        running_summary = strip_think_tags(str(ext_response.content))

    logger.info("Reflection complete")
    return {
        "running_summary": running_summary,
        "data_research_results": data_results,
        "all_tool_messages": all_msgs,
    }


# ── Node 5: finalize_summary（对应 AIRA finalize_summary）────────────────

def finalize_summary(state: RCAState, config: RunnableConfig) -> dict:
    """
    对应 AIRA 的 finalize_summary 节点（nodes.py:finalize_summary）。
    将 RCA 分析转换为 CausalGraph JSON 输出。

    AIRA 原始流程：
      running_summary → finalize_report prompt → final markdown + sources
    RCA 替换：
      running_summary → compress prompts (from RolloutRunner) → CausalGraph JSON
    """
    logger.info("FINALIZING REPORT")

    # 使用 RolloutRunner 传入的 compress prompts 生成最终输出
    compress_sp = config["configurable"]["compress_system_prompt"]
    compress_up = config["configurable"]["compress_user_prompt"]
    llm = _make_model(max_tokens=32000)

    running_summary = state.get("running_summary") or ""

    messages = [
        SystemMessage(content=compress_sp),
        HumanMessage(
            content=(
                f"Here is my complete RCA analysis:\n\n"
                f"{running_summary}\n\n"
                f"{compress_up}"
            )
        ),
    ]
    response = llm.invoke(messages)

    logger.info("Finalization complete")
    return {"final_report": str(response.content)}


# ── Build Graph（保留 AIRA 的多阶段流水线拓扑）────────────────────────────

def build_agent():
    """
    保留 AIRA 的多阶段流水线拓扑。

    AIRA 原始图：
      Stage 1 (generate_queries): START → generate_query → END
      Stage 2 (generate_summary): START → web_research → summarize_sources
                                        → reflect_on_summary → finalize_summary → END

    RCA 合并图（Stage 1 + Stage 2 合为单图）：
      START → generate_queries → data_research → summarize_sources
            → reflect_on_summary → finalize_summary → END
    """
    builder = StateGraph(RCAState)

    # Stage 1 节点（对应 AIRA generate_queries 图）
    builder.add_node("generate_queries", generate_queries)

    # Stage 2 节点（对应 AIRA generate_summary 图）
    builder.add_node("data_research", data_research)          # 替代 web_research
    builder.add_node("summarize_sources", summarize_sources)
    builder.add_node("reflect_on_summary", reflect_on_summary)
    builder.add_node("finalize_summary", finalize_summary)

    # 边（保留 AIRA 的线性流水线拓扑）
    builder.add_edge(START, "generate_queries")
    builder.add_edge("generate_queries", "data_research")
    builder.add_edge("data_research", "summarize_sources")
    builder.add_edge("summarize_sources", "reflect_on_summary")
    builder.add_edge("reflect_on_summary", "finalize_summary")
    builder.add_edge("finalize_summary", END)

    return builder.compile()


# ── 工具函数 ─────────────────────────────────────────────────────────────

def strip_markdown_json(text: str) -> str:
    """剥离 LLM 返回的 ```json ... ``` 代码块，提取纯 JSON。"""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


# ── LangChain → OpenAI 格式转换（与 thinkdepthai 相同）────────────────────

def to_openai_message(msg) -> dict | None:
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": str(msg.content)}

    if isinstance(msg, AIMessage):
        tool_calls = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["args"], ensure_ascii=False),
                },
            }
            for tc in (msg.tool_calls or [])
        ]
        # ChatAnthropic returns content as list of blocks; extract text parts
        content = msg.content
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) and b.get("type") == "text" else ""
                for b in content
            ).strip()
        entry: dict = {
            "role": "assistant",
            "content": str(content) if content else "",
        }
        if tool_calls:
            entry["tool_calls"] = tool_calls
        return entry

    if isinstance(msg, ToolMessage):
        return {
            "role": "tool",
            "content": str(msg.content),
            "tool_call_id": msg.tool_call_id,
        }

    return None


def convert_trajectory(messages: list) -> list[dict]:
    return [m for msg in messages if (m := to_openai_message(msg)) is not None]


# ── 主流程（RolloutRunner stdin/stdout 接口）─────────────────────────────

def main():
    payload = json.loads(sys.stdin.read())

    data_dir = payload.get("data_dir", "")

    # 用 data_dir 增强 question（与 thinkdepthai 一致：将数据位置追加到问题中）
    question = payload.get("question", "")
    if data_dir:
        question = (
            f"{question}\n\n## Data Location\n\n"
            f"The telemetry data for this incident is located at: `{data_dir}`\n"
            f"Start by calling `list_tables_in_directory(directory=\"{data_dir}\")` "
            f"to discover available parquet files."
        )

    # 构建 config（对应 AIRA 的 config["configurable"]）
    config = {
        "configurable": {
            "question": question,                       # augmented question（增强后）
            "user_prompt": payload["user_prompt"],
            "system_prompt": payload["system_prompt"],
            "data_dir": data_dir,
            "compress_system_prompt": payload["compress_system_prompt"],
            "compress_user_prompt": payload["compress_user_prompt"],
            "number_of_queries": 1,   # 生成 1 个综合调查查询
            "num_reflections": 1,     # 1 轮反思（对应 AIRA reflection_count）
        }
    }

    agent = build_agent()

    # 初始状态（对应 AIRA 的 input={"queries": [], "web_research_results": [], "running_summary": ""}）
    initial_state = {
        "queries": [],
        "data_research_results": [],
        "running_summary": "",
        "final_report": "",
        "all_tool_messages": [],
    }

    # 运行 AIRA 多阶段流水线
    result_state = agent.invoke(input=initial_state, config=config)

    # 输出结果
    output = strip_markdown_json(result_state.get("final_report", ""))
    all_tool_msgs = result_state.get("all_tool_messages", [])
    trajectory = convert_trajectory(all_tool_msgs)

    # Usage 采集：优先用 UsageTracker（monkey-patch OpenAI/Anthropic SDK），
    # 非 Claude 模型走 ChatOpenAI SDK，install_openai_hooks 可拦截
    usage = _tracker.get_usage()

    result = {"output": output, "trajectory": trajectory, "usage": usage}
    # 单行输出，runner._parse_last_json 从末行解析
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
