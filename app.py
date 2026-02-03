# app.py
# import os
# import json
# from dotenv import load_dotenv
# from typing import Dict, Any
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.agents import create_agent
# from langchain.agents.middleware import wrap_tool_call
# from langchain.messages import ToolMessage

# from tools.langchain_tools_test import TOOLS, _get_data

# load_dotenv()

# @wrap_tool_call
# def handle_tool_errors(request, handler):
#     try:
#         return handler(request)
#     except Exception as e:
#         return ToolMessage(
#             content=f"Tool error: {str(e)}",
#             tool_call_id=request.tool_call.get("id")
#         )

# def build_llm():
#     model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
#     return ChatGoogleGenerativeAI(model=model_name, temperature=0.5)

# def build_agent():
#     _get_data()  # warm-load once
#     llm = build_llm()
#     return create_agent(
#         llm,
#         tools=TOOLS,
#         middleware=[handle_tool_errors],
#         system_prompt=(
#             """
#                 You are a merchant analytics assistant for the Olist e-commerce dataset.
#                 Your job is to answer questions with correct numbers by calling tools.

#                 You have tools that compute demand using reliable joins:
#                 - get_data_dictionary: learn tables, columns, join keys, and metric definitions.
#                 - top_demand_products: top products by units sold (joins orders + order_items + products + translation).
#                 - demand_by_category: top categories by units sold.
#                 - product_demand_summary: summary demand for one product_id.
#                 - get_daily_demand: daily time series demand for one product/category.
#                 - compute_product_unpredictability: volatility score for products.
#                 - classify_demand_types: smooth/erratic/intermittent/lumpy classification.

#                 Rules:
#                 1) NEVER guess numeric answers. Use tools.
#                 2) If the user asks “top / most / highest demand”, use top_demand_products or demand_by_category.
#                 3) If the user asks demand for a specific product_id, use product_demand_summary first.
#                 If they ask for the time series trend, then use get_daily_demand.
#                 4) When a request is ambiguous (product vs category vs date range), ask ONE short follow-up question.
#                 5) Always define 'demand' as: count of rows in order_items for delivered orders (unless delivered_only=False).
#                 6) Keep outputs short: show top-k and include what filters you applied (delivered_only, date range).
#                 """
#         ),
#     )

# def run_repl():
#     if not os.getenv("GEMINI_API_KEY"):
#         raise RuntimeError("GEMINI_API_KEY not set in environment (.env or shell).")

#     os.environ.setdefault("OLIST_DATA_DIR", "./olist_data")

#     agent = build_agent()
#     print("✅ Olist Agent (Gemini + LangChain create_agent)")
#     print("Type 'exit' to quit.\n")

#     while True:
#         user = input("You> ").strip()
#         if not user:
#             continue
#         if user.lower() in {"exit", "quit"}:
#             break

#         request = {"messages": [{"role": "user", "content": user}]}
#         try:
#             result = agent.invoke(request)
#             print("\n[raw]\n", json.dumps(result, indent=2, default=str))
#             if isinstance(result, dict) and "output" in result:
#                 print("\nAgent>\n", result["output"], "\n")
#             else:
#                 print("\nAgent>\n", result['messages'][:]['content'], "\n")
#         except Exception as e:
#             print("\n[Error calling agent]:", e, "\n")




# if __name__ == "__main__":
#     run_repl()


# app.py
import os
import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

# import your tools and data-warm function (adjust module name if needed)
from tools.langchain_tools_test import TOOLS, _get_data

load_dotenv()


# ---------- middleware to catch tool errors and return a ToolMessage ----------
@wrap_tool_call
def handle_tool_errors(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: {str(e)}",
            tool_call_id=request.tool_call.get("id")
        )


# ---------- LLM + agent builder ----------
def build_llm():
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    return ChatGoogleGenerativeAI(model=model_name, temperature=0.2)


def build_agent():
    # warm-load dataset once
    _get_data()
    llm = build_llm()
    system_prompt = (
        "You are a merchant analytics assistant for the Olist e-commerce dataset.\n"
        "Your job is to answer questions with correct numbers by calling tools.\n\n"
        "Tools available:\n"
        "- get_data_dictionary: learn tables, columns, join keys, and metric definitions.\n"
        "- top_demand_products: top products by units sold (joins orders + order_items + products + translation).\n"
        "- demand_by_category: top categories by units sold.\n"
        "- product_demand_summary: summary demand for one product_id.\n"
        "- get_daily_demand: daily time series demand for one product/category.\n"
        "- compute_product_unpredictability: volatility score for products.\n"
        "- classify_demand_types: smooth/erratic/intermittent/lumpy classification.\n\n"
        "Rules (follow these exactly):\n"
        "1) NEVER guess numeric answers. Use tools for counts/aggregates.\n"
        "2) If the user asks 'top / most / highest demand', prefer top_demand_products or demand_by_category.\n"
        "3) For product-specific demand use product_demand_summary first, then get_daily_demand for time series.\n"
        "4) Ask ONE short clarifying question if the user's request is ambiguous (product vs category vs date range).\n"
        "5) Define 'demand' as COUNT(order_items rows) for delivered orders unless the user requests otherwise.\n"
        "6) Keep outputs short and include the filters applied (delivered_only, date range) when returning numbers.\n"
    )

    agent = create_agent(
        llm,
        tools=TOOLS,
        middleware=[handle_tool_errors],
        system_prompt=system_prompt
    )
    return agent


# ---------- helpers to extract message contents ----------
def _normalize_content_field(content: Any) -> List[str]:
    """Normalize a message content into a list of text strings."""
    if content is None:
        return []
    if isinstance(content, str):
        return [content.strip()] if content.strip() else []
    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    texts.append(str(part["text"]).strip())
            elif isinstance(part, str):
                if part.strip():
                    texts.append(part.strip())
        return texts
    # fallback: convert to string
    try:
        s = str(content)
        return [s] if s else []
    except Exception:
        return []


def extract_all_messages_content(result: Dict[str, Any]) -> List[str]:
    """
    Given the agent.invoke() result (a dict), extract textual content from every message
    in result['messages'] preserving order. Works for dict messages and LangChain message objects.
    """
    if not isinstance(result, dict):
        return []

    messages = result.get("messages", [])
    if not isinstance(messages, list):
        return []

    out_texts: List[str] = []
    for m in messages:
        # dict-like message
        if isinstance(m, dict):
            content = m.get("content") or m.get("text") or m.get("output")
            out_texts.extend(_normalize_content_field(content))
            continue

        # object-like message (AIMessage, ToolMessage, etc.)
        if hasattr(m, "content"):
            content = getattr(m, "content")
            out_texts.extend(_normalize_content_field(content))
            continue

        # fallback: string
        if isinstance(m, str):
            s = m.strip()
            if s:
                out_texts.append(s)

    return out_texts


# ---------- REPL ----------
def run_repl():
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY not set in environment (.env or shell).")

    os.environ.setdefault("OLIST_DATA_DIR", "./olist_data")

    agent = build_agent()
    print("✅ Olist Agent (Gemini + LangChain create_agent)")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user = input("You> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break

        # Build the request dict (must be a dict for agent.invoke)
        request = {"messages": [{"role": "user", "content": user}]}

        try:
            result = agent.invoke(request)

            # Extract and print all message contents (agent decisions, tool outputs, final answer)
            contents = extract_all_messages_content(result)
            if contents:
                for i, txt in enumerate(contents, start=1):
                    print(f"\n--- message {i} ---\n{txt}\n")
            else:
                print("\n(no textual messages found)\n")

            # Optional: print raw trace if DEBUG_RAW=1
            if os.getenv("DEBUG_RAW") == "1":
                print("\n[raw]\n", json.dumps(result, indent=2, default=str))

        except Exception as e:
            print("\n[Error calling agent]:", e, "\n")


if __name__ == "__main__":
    run_repl()
