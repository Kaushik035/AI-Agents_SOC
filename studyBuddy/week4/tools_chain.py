"""
This module contains:

1.  Base tool wrappers
    • Tavily web search
    • Wikipedia two-sentence summary
    • Safe AST calculator

2.  Validation helpers for wiki + calculator results

3.  Intent detection (decides which chain to run)

4.  Sequential tool-chaining engine
    • Handles simple look-ups
    • Handles arithmetic expressions
    • Handles ‘lookup-then-calculate’ cases
    • Provides error-recovery / fallback

5.   last-resort OpenAI fallback

The chatbot imports **run_tool_chain(query)** and drops the returned
string into the final prompt.
"""

# ──────────────────────────────────────────────
#  Imports & environment
# ──────────────────────────────────────────────
from __future__ import annotations

import os
import re
import ast
import operator
from typing import Tuple

import requests
import wikipedia
from dotenv import load_dotenv

# OpenAI fallback (only used when everything else fails)
from shared.newOpenAI import openai

load_dotenv()
TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")


# ──────────────────────────────────────────────
#  1.  Base Tool Wrappers
# ──────────────────────────────────────────────
def tavily_search(query: str) -> str:
    """
    Return a quick answer from Tavily.

    If the API fails or times out, an error string is returned instead.
    """
    url: str = "https://api.tavily.com/search"
    payload: dict = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "include_answer": True,
    }

    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code == 200:
            print(f"Tavily response: {response.json()}")
            return response.json().get("answer", "No answer found.")
        print(f"Tavily error: {response.status_code} - {response.text}")
        return f"Tavily error ({response.status_code})"
    except Exception as exc:  # noqa: BLE001
        return f"Tavily exception: {exc}"


def wiki_summary(query: str) -> str:
    """
    Return the first two sentences of a Wikipedia article
    or an error string if not available.
    """
    try:
        return wikipedia.summary(query, sentences=2)
    except wikipedia.exceptions.DisambiguationError as exc:
        return f"Multiple options found: {exc.options[:5]}"
    except wikipedia.exceptions.PageError:
        return "No page found."
    except Exception as exc:  # noqa: BLE001
        return f"Wikipedia exception: {exc}"


def safe_calculate(expression: str) -> str:
    """
    Evaluate a basic arithmetic expression safely using Python’s AST.

    Supported operators: +, -, *, /, ** (power)
    """
    try:
        node = ast.parse(expression, mode="eval")
        result = _eval_ast(node.body)
        return str(result)
    except Exception:  # noqa: BLE001
        return "Calculation error."


def _eval_ast(node):
    """Recursive helper for safe_calculate()."""
    if isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return node.n

    if isinstance(node, ast.BinOp):  # type: ignore[attr-defined]
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)

        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
        }
        return ops[type(node.op)](left, right)

    raise ValueError("Unsupported expression")


# ──────────────────────────────────────────────
#  2.  Validation Helpers
# ──────────────────────────────────────────────
def validate_wiki(text: str) -> bool:
    """Return True if the wiki_summary() output looks usable."""
    bad_phrases = ["No page found", "Multiple options", "exception"]
    return len(text.split()) >= 10 and not any(b in text for b in bad_phrases)


def validate_calc(text: str) -> bool:
    """Return True if the calculator result does not contain 'error'."""
    return "error" not in text.lower()


# ──────────────────────────────────────────────
#  3.  Utility Helper
# ──────────────────────────────────────────────
def extract_first_number(text: str) -> str | None:
    """
    Extract the first numeric token from text (commas are stripped).

    Example:
        "Paris has about 2,123,000 inhabitants" → "2123000"
    """
    match = re.search(r"\d[\d,\.]*", text)
    return match.group(0).replace(",", "") if match else None


# ──────────────────────────────────────────────
#  4.  Intent Detection
# ──────────────────────────────────────────────
def detect_intent(query: str) -> str:
    """
    Returns one of:  search / wiki / calculator / calc_with_lookup / none
    """
    q = query.lower()

    if any(k in q for k in ("latest", "search", "recent", "find", "look up")):
        return "search"

    if any(k in q for k in ("wiki", "who is", "who was")):
        return "wiki"

    if any(k in q for k in ("calculate", "times", "multiplied by")) or re.search(r"\d\s*[\+\-\*\/]\s*\d", q):
        # Could be “just math” or “math + real-world number”
        if any(
            kw in q
            for kw in ("population", "gdp", "area", "height", "length", "distance")
        ):
            return "calc_with_lookup"
        return "calculator"

    return "none"


# ──────────────────────────────────────────────
#  5.  Sequential Tool-Chaining Engine
# ──────────────────────────────────────────────

def run_tool_chain(query: str) -> str:
    """
    Execute an appropriate tool chain based on the detected intent.

    The function always returns a **string** (even when an error occurs)
    so the chatbot can inject it into the final LLM prompt.
    """

    intent = detect_intent(query)
    print(f"Detected intent: {intent}")

    # 5-a : Simple web search
    if intent == "search":
        try:
            answer = tavily_search(query)
            return answer
        except Exception as e:
            return f"[Tavily failed: {e}] " + fallback_openai(query)

    # 5-b : Wikipedia summary with validation & fallback
    if intent == "wiki":
        try:
            wiki = wiki_summary(query)
            if validate_wiki(wiki):
                return wiki

            # wiki failed → try Tavily as backup
            try:
                return tavily_search(query)
            except Exception as e:
                return f"[Wiki and Tavily failed: {e}] " + fallback_openai(query)
        except Exception as e:
            return f"[Wiki failed: {e}] " + fallback_openai(query)

    # 5-c : Pure arithmetic calculation
    if intent == "calculator":
        try:
            # strip the word “calculate” if present
            expression = (
                query.split("calculate", 1)[-1].strip()
                if "calculate" in query.lower()
                else query
            )
            return safe_calculate(expression)
        except Exception as e:
            return f"[Calculator failed: {e}] " + fallback_openai(query)

    # 5-d : Lookup then calculate  (dependency chain)
    if intent == "calc_with_lookup":
        try:
            multiplier_match = re.search(r"\b\d[\d,\.]*\b", query)
            multiplier = (
                multiplier_match.group(0).replace(",", "") if multiplier_match else None
            )

            fact_match = re.search(
                r"(population|gdp|area|height|length|distance)\s+of\s+([\w\s]+)",
                query.lower(),
            )
            if fact_match:
                fact_type = fact_match.group(1)
                fact_target = fact_match.group(2).strip()
            else:
                fact_type = fact_target = None

            if not (multiplier and fact_type and fact_target):
                return "Sorry, I couldn’t parse the calculation request."

            wiki_query = f"{fact_type} of {fact_target}"
            wiki_result = wiki_summary(wiki_query)

            if not validate_wiki(wiki_result):
                return f"Could not fetch {fact_type} data for {fact_target}."

            fact_number = extract_first_number(wiki_result)
            if not fact_number:
                return f"No numeric {fact_type} value found in Wikipedia summary."

            calculation = f"{multiplier} * {fact_number}"
            calc_result = safe_calculate(calculation)

            if not validate_calc(calc_result):
                return "Calculation error."

            return f"{multiplier} × {fact_number} = {calc_result}"

        except Exception as e:
            return f"[Calc-with-lookup failed: {e}] " + fallback_openai(query)

    # 5-e : No intent matched  → fallback directly
    return fallback_openai(query)



# ──────────────────────────────────────────────
#  6.  Last-resort OpenAI fallback
# ──────────────────────────────────────────────
def fallback_openai(query: str) -> str:
    """Only used if you explicitly want an LLM backup inside tools.py."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": query}],
    )
    return response["choices"][0]["message"]["content"]
