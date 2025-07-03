"""
• System Prompt Engineering (base + domain + style)
• Dynamic Style Conditioning (auto user-level inference)
• Enhanced Domain Detection (embeddings + keywords)
• Ethical Guard-Rails (Detoxify fallback, file logging)
"""

import os
from pathlib import Path
from typing import Tuple

#  1.  System-Prompt Base
SYSTEM_PROMPT_BASE = (
    "You are Study Buddy, a friendly and knowledgeable peer tutor. "
    "Your goal is to help students understand academic concepts in a clear, "
    "concise, and engaging way. Always respond with a supportive tone, "
    "breaking down complex ideas into simple explanations. Use examples when "
    "helpful and avoid jargon unless explained. If unsure, admit it and "
    "suggest how to find the answer. Prioritize accuracy, clarity, and avoid "
    "biased or harmful content. Avoid controversial or offensive language."
)

#  2.  Domain-Specific Prompts
DOMAIN_PROMPTS = {
    "computer_science": """
You are an expert in computer science. Use terms like 'algorithm',
'data structure', or 'runtime complexity' accurately. Provide code
snippets when relevant, and avoid oversimplifying technical details.
""",
    "biology": """
You are a biology tutor. Use terms like 'cell', 'ecosystem', or 'DNA'
accurately. Include biological processes and concrete examples, such as
how enzymes work or species interactions.
""",
    "physics": """
You are a physics tutor. Use correct terminology such as 'momentum',
'quantum', 'energy', and 'wavefunction'. Provide clear derivations or
thought experiments when helpful.
""",
    "history": """
You are a history tutor. Use accurate dates, events, and figures.
Provide context for historical events and avoid present-day bias.
""",
}

DOMAIN_KEYWORDS = {
    "computer_science": ["algorithm", "data structure", "python", "code", "programming", "database"],
    "biology": ["cell", "dna", "enzyme", "photosynthesis", "evolution", "plant"],
    "physics": ["quantum", "entropy", "momentum", "relativity", "electron"],
    "history": ["war", "revolution", "empire", "ancient", "dynasty", "medieval"],
}

#  embedding-based detection
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    _domain_labels = list(DOMAIN_PROMPTS.keys())  # ⟵ ['computer_science', ...]
    _domain_vectors = _embedder.encode(_domain_labels, convert_to_numpy=True)
except Exception:
    _embedder = None
    _domain_vectors = None


def _domain_by_embedding(query: str) -> str:
    if _embedder is None:
        print("Embedding model not available, falling back to default domain.")
        return "default"
    vec = _embedder.encode([query], convert_to_numpy=True)[0]

    # sims is a vector of cosine similarities between the query and each domain.
    sims = np.dot(_domain_vectors, vec) / (
        np.linalg.norm(_domain_vectors, axis=1) * np.linalg.norm(vec) + 1e-8
    )
    # print(f"Embedding similarities: {sims}")
    best_idx = int(np.argmax(sims))
    if sims[best_idx] > 0.20:  # heuristic threshold
        return _domain_labels[best_idx]
    return "default"


def detect_domain(query: str) -> str:
    """Return domain key using embeddings first, then keywords."""
    q_lower = query.lower()

    #1 embedding similarity
    domain = _domain_by_embedding(q_lower)
    if domain != "default":
        return domain

    #2 keyword fallback if embedding fails
    for dom, kws in DOMAIN_KEYWORDS.items():
        if any(kw in q_lower for kw in kws):
            return dom

    return "default"


#  3.  Response-Style Conditioning
STYLE_GUIDELINES = {
    "high_school": (
        "Use simple words and a casual, friendly tone—as if explaining to a "
        "classmate. Include fun analogies."
    ),
    "college": (
        "Use precise terminology and a professional yet approachable tone. "
        "Provide detailed examples and, when helpful, formal definitions."
    ),
    "professional": (
        "Use domain-appropriate technical language and concisely reference "
        "formulas, research papers, or industry standards when relevant."
    ),
}


def infer_user_level(query: str) -> str:
    """
    Very naive heuristic: detect if query contains advanced vocabulary or
    cites research; can be replaced with a classifier later.
    """
    q = query.lower()
    if any(word in q for word in ["prove", "theorem", "derive", "complexity"]):
        return "college"
    if any(word in q for word in ["peer-review", "industrial", "research"]):
        return "professional"
    return "high_school"


def build_persona_system_prompt(query: str, user_level: str | None = None) -> str:
    """
    Combine:
      • global base prompt
      • domain-specific block
      • style block (auto-inferred if None)
    """
    domain = detect_domain(query)
    print(f"Detected domain: {domain} ")
    domain_block = DOMAIN_PROMPTS.get(domain, "")
    level = user_level or infer_user_level(query)
    # print(f"Inferred user level: {level} ")
    style_block = STYLE_GUIDELINES.get(level, STYLE_GUIDELINES["high_school"])

    return f"{SYSTEM_PROMPT_BASE}\n{domain_block}\nAdditional instructions: {style_block}"


#  4.  Ethical Guard-Rails

#1 Try Detoxify → else fallback to rule-based
try:
    from detoxify import Detoxify

    _detox = Detoxify("original-small")
except Exception:
    _detox = None

SENSITIVE_TERMS = [
    "hate",
    "violence",
    "discriminate",
    "offensive",
]

# Logs problematic responses to a file (guardrail_log.txt) for auditing or improvement later.
def _log_flag(msg: str, text: str) -> None:
    """Append flagged text to `guardrail_log.txt`."""
    log_file = Path("guardrail_log.txt")
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"[{msg}]\n{text}\n{'-'*60}\n")


def check_ethical_compliance(text: str) -> Tuple[bool, str]:
    """
    Return (is_compliant, message). Uses Detoxify if available, otherwise
    falls back to simple term/bias checks.
    """
    if _detox:
        scores = _detox.predict(text)
        print(f"Detoxify scores: {scores.get('toxicity', 0):.2f}")
        if scores.get("toxicity", 0) > 0.45:
            _log_flag("Detoxify toxicity > 0.4", text)
            return False, "Potentially toxic content"

    tl = text.lower()
    for term in SENSITIVE_TERMS:
        if term in tl:
            _log_flag(f"Sensitive term: {term}", text)
            return False, f"Response contains sensitive term: {term}"

    if any(phrase in tl for phrase in ["better than", "superior race", "inferior"]):
        _log_flag("Bias phrase detected", text)
        return False, "Potential bias detected"

    return True, "Compliant"
