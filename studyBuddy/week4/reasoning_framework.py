"""
Reasoning Frameworks for Study Buddy

• Plan-Execute-Refine  (generic, any subject)
• Self-Correction      (LLM reviews & fixes its own answer)
• Confidence Scoring   (choose best of multiple candidates)

"""

from __future__ import annotations
from typing import List, Tuple
import re

from shared.newOpenAI import openai
from persona import build_persona_system_prompt, check_ethical_compliance


#  0.  Low-level OpenAI wrapper (no persona injected here)
def _chat(messages: List[dict]) -> str:
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    return resp["choices"][0]["message"]["content"]

#  1.  Plan-Execute-Refine (PER) — generic version
def plan_execute_refine(
    query: str, context_msgs: List[dict], user_level: str = "high_school"
) -> Tuple[str, str]:
    """
    Returns (final_answer, raw_plan).  Works for *most* open-ended or procedural
    questions — maths, coding, science derivations, etc.
    """
    sys_prompt = build_persona_system_prompt(query, user_level)

    #1 PLAN
    plan_prompt = (
        "Outline a step-by-step plan to answer the following question. "
        "Return the plan as a numbered list.\n"
        f"Question: {query}"
    )
    plan = _chat([{"role": "system", "content": sys_prompt},
                  {"role": "system", "content": plan_prompt}, *context_msgs])

    #2 EXECUTE
    exec_prompt = (
        f"Follow this plan to answer the question.\nPlan:\n{plan}\n\n"
        "Provide a detailed answer. Show calculations or reasoning openly."
    )
    execution = _chat([{"role": "system", "content": sys_prompt},
                       {"role": "system", "content": exec_prompt}, *context_msgs])

    #3 REFINE
    refine_prompt = (
        "Review the answer for accuracy, logic and clarity. "
        "Correct any mistake and rewrite succinctly:\n"
        f"{execution}"
    )
    refined = _chat([{"role": "system", "content": sys_prompt},
                     {"role": "system", "content": refine_prompt}, *context_msgs])

    return refined, plan


#  2.  Self-correction
def self_correct_response(
    query: str, draft: str, context_msgs: List[dict], user_level: str = "high_school"
) -> str:
    """
    Ask the LLM to check its own answer and fix errors.
    """
    sys_prompt = build_persona_system_prompt(query, user_level)

    detect_prompt = (
        "Review the following answer for any factual, logical, or numerical errors. "
        "If none, reply 'No errors detected'.  Otherwise list the issues.\n\n"
        f"Q: {query}\nA: {draft}"
    )
    report = _chat([{"role": "system", "content": sys_prompt},
                    {"role": "system", "content": detect_prompt}, *context_msgs])

    if "no errors detected" in report.lower():
        return draft

    correct_prompt = (
        "Based on these issues, rewrite a *corrected* answer:\n\n"
        f"Issues:\n{report}\n\nOriginal answer:\n{draft}"
    )
    return _chat([{"role": "system", "content": sys_prompt},
                  {"role": "system", "content": correct_prompt}, *context_msgs])


#  3.  Confidence scoring & selection

from sentence_transformers import SentenceTransformer, util
import re
import numpy as np

_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_VAGUE_PHRASES = re.compile(
    r"(i (am|\'m) not sure|cannot find|no answer|maybe|might be)", re.I
)


def _confidence(
    resp: str,
    query: str,
    *,
    origin: str = "LLM",
    conv_context: str = ""
) -> float:
    """
    Composite score ∈ [0‥1].

       • 35 % semantic sim  (answer ↔ query)
       • 15 % semantic sim  (answer ↔ recent-context)
       • 25 % keyword overlap (answer ↔ query)
       • 15 % ‘reasonable length’ bonus   (50–250 words ⇢ full credit)
       • 10 % clarity / no-red-flags bonus
    """
    score = 0.0

    #  1-a  answer ↔ query  (25 %)
    q_emb  = _embedder.encode(query,        convert_to_tensor=True, normalize_embeddings=True)
    a_emb  = _embedder.encode(resp,         convert_to_tensor=True, normalize_embeddings=True)
    sim_q  = float(util.cos_sim(q_emb, a_emb)[0][0])
    score += 0.25 * max(sim_q, 0.0)

    # 1-b  answer ↔ convo-context  (25 %)
    if conv_context:
        c_emb  = _embedder.encode(conv_context, convert_to_tensor=True, normalize_embeddings=True)
        sim_c  = float(util.cos_sim(c_emb, a_emb)[0][0])
        score += 0.25 * max(sim_c, 0.0)

    # 2  keyword overlap  (25 %)
    q_set = set(re.findall(r"\w+", query.lower()))
    a_set = set(re.findall(r"\w+", resp.lower()))
    kv_overlap = len(q_set & a_set) / max(len(q_set), 1)
    score += 0.25 * kv_overlap

    # 3  length bonus  (15 %)
    words = len(resp.split())
    if 50 <= words <= 250:
        length_bonus = 1.0
    else:
        length_bonus = max(0.0, 1 - abs(words - 150) / 300)   # linear fall-off
    score += 0.15 * length_bonus

    # 4  clarity / no red-flags  (10 %)
    if not _VAGUE_PHRASES.search(resp) and "error" not in resp.lower():
        score += 0.10

    return round(min(score, 1.0), 3)



#  LLM-based tie-breaker (quick rubric 0–10)
def _llm_grade(query: str, answer: str) -> float:
    """
    Ask GPT to grade the answer quality. Returns a 0-10 float.
    Uses a *very* short prompt so it stays cheap.
    """
    prompt = (
        "You are an impartial evaluator.\n"
        f"Q: {query}\nA: {answer}\n\n"
        "Give a single line with ONLY a score from 0 (terrible) to 10 (perfect)."
    )
    try:
        g = _chat([{"role": "system", "content": prompt}]).strip()
        return max(0.0, min(10.0, float(g)))
    except Exception:
        # fall back to neutrality if something odd happens
        return 5.0



_TOOL_LIKE = {"Tool-only", "Tavily", "Wikipedia", "Calculator"}

def _needs_self_correction(
    query: str,
    answer: str,
    origin: str = "LLM",          # <–– pass the label
) -> bool:
    """
    Return True only when self-check is valuable.

    • Skip for short tool answers (<= 80 words)  
    • Otherwise use the old heuristics.
    """
    # ── 0. Short, tool-sourced answers are assumed trustworthy
    if origin in _TOOL_LIKE and len(answer.split()) <= 80:
        return False

    q = query.lower()
    trig = (
        "+", "-", "*", "/", "integral", "solve", "equation",
        "derive", "proof",
    )
    if any(w in q for w in trig):
        return True
    if len(answer.split()) > 120:
        return True
    if re.search(r"\d", answer):
        return True
    return False






#  4.  Public helper – produce final answer with reasoning
def reasoned_answer(
    query: str,
    context_msgs: List[dict],
    rag_notes: str,
    tool_output: str | None = None,
    user_level: str = "high_school",
) -> str:
    """
    Build several candidate answers, score them, pick the best,
    run self-correction and ethical checks.
    """
    sys_prompt = build_persona_system_prompt(query, user_level)

    # ── build candidate list 
    candidates: List[Tuple[str, str]] = []

    # A) RAG + Tool evidence
    evidence_block = (
        f"Relevant notes:\n{rag_notes}\n\n"
        f"External tool output:\n{tool_output or '—'}\n\n"
        "The tool output is very important to take care of when the user is asking questions about some latest news or to search in the web.\n\n"
        "Answer the user's question clearly."
        
    )
    evidence_ans = _chat(
        [{"role": "system", "content": sys_prompt},
         {"role": "system", "content": evidence_block}, *context_msgs,
         {"role": "user",   "content": query}]
    )
    candidates.append(("RAG+Tool", evidence_ans))

    # B) Plan-Execute-Refine  (procedural queries)
    if any(k in query.lower() for k in ("how", "why", "solve", "derive", "calculate")):
        per_ans, _ = plan_execute_refine(query, context_msgs, user_level)
        candidates.append(("Plan-Execute-Refine", per_ans))

    # C) Tool-only  (if non-trivial)
    if tool_output and tool_output.strip().lower() not in (
        "no external tool used.", "no answer found."
    ):
        candidates.append(("Tool-only", tool_output))

    # D) General LLM  (always include)
    general_ans = _chat(
        [{"role": "system", "content": sys_prompt}, *context_msgs,
         {"role": "user",   "content": query}]
    )
    candidates.append(("General-LLM", general_ans))

    # just before the scoring loop
    conv_ctx_text = "\n".join(m["content"] for m in context_msgs)  # recent+relevant slice

    # replace the list-comprehension that builds `scored`
    scored = [
        (
            label,
            ans,
            _confidence(
                ans,
                query,
                origin=label,
                conv_context=conv_ctx_text      # NEW: adds context similarity
            ),
        )
        for label, ans in candidates
    ]

    print("Candidate scores:", scored)
    # highest heuristic score
    best_label, best_ans, best_score = max(scored, key=lambda t: t[2])



    print(f"Best candidate: {best_label} (score {best_score:.2f})")

    if "i don't have real-time" in best_ans.lower():
        print("Primary answer lacks real-time info. Falling back...")
        fallback = sorted(
            [t for t in scored if t[0] != best_label],
            key=lambda t: t[2],
            reverse=True
        )
        if fallback:
            fb_label, fb_ans, fb_score = fallback[0]
            best_label, best_ans, best_score = fb_label, fb_ans, fb_score
            print(f"Fallback used: {fb_label} (score {fb_score:.2f})")

    print("Best answer:", best_ans[:100] + "...")  # show first 100 chars
    # ── OPTIONAL TIE-BREAKER when tool-only is on top
    if best_label in {"Tool-only", "Tavily"}:
        # find best *non-tool* candidate for comparison
        print("Tool-based answer selected, checking alternatives...")
        non_tool = sorted(
            [t for t in scored if t[0] not in {"Tool-only", "Tavily"}],
            key=lambda t: t[2],
            reverse=True,
        )
        if non_tool:
            alt_label, alt_ans, _ = non_tool[0]

            tool_grade = _llm_grade(query, best_ans)
            alt_grade  = _llm_grade(query, alt_ans)

            # Debug:
            # print(f"LLM grades — tool:{tool_grade:.1f}  alt:{alt_grade:.1f}")

            if alt_grade > tool_grade + 1:          # +1 ⇒ avoid flip-flop
                best_label, best_ans, best_score = (
                    alt_label,
                    alt_ans,
                    _confidence(alt_ans, query, origin=alt_label),
                )


    # ── self-correct
    if _needs_self_correction(query, best_ans, origin=best_label):
        print("Self-correction needed for:", best_label)
        corrected = self_correct_response(query, best_ans, context_msgs, user_level)
    else:
        corrected = best_ans

    # ── ethical guard-rail
    compliant, msg = check_ethical_compliance(corrected)
    if not compliant:
        return (
            "⚠️ Sorry, I can’t provide that response due to ethical concerns "
            f"({msg}). Please rephrase your question."
        )

    return (
        f"Response (via {best_label}, confidence {best_score:.2f}):\n"
        f"{corrected}"
    )

