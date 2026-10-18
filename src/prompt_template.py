# src/prompt_template.py — Unified Prompt Formatting for Training and Evaluation
import pandas as pd
from typing import Optional, Tuple, Dict

MAX_BODY_TOKENS = 800
RESPONSE_TEMPLATE = "### Response:\n"

def truncate_body_text(body: str, max_tokens: int = MAX_BODY_TOKENS, tokenizer = None) -> Tuple[str, bool]:
    """
    Truncates email body text to a fixed token budget BEFORE formatting.
    If a tokenizer is provided, tokenizes and slices. Otherwise uses word-level approximation.
    Returns (truncated_body, was_truncated).
    """
    if not body or not isinstance(body, str):
        return "", False
    
    if tokenizer is not None:
        tokens = tokenizer.encode(body, add_special_tokens=False)
        if len(tokens) > max_tokens:
            truncated = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
            return truncated, True
        return body, False
    else:
        # Approximate 1 token ~= 0.75 words (or ~1.3 tokens per word)
        words = body.split()
        max_words = int(max_tokens * 0.75)
        if len(words) > max_words:
            return " ".join(words[:max_words]), True
        return body, False

def build_naive_prompt(subject: str, body: str) -> str:
    """
    Constructs the prompt up to '### Response:\n' (used for evaluation & as 'prompt' column).
    """
    subject_str = str(subject) if pd.notna(subject) else ""
    body_str = str(body) if pd.notna(body) else ""

    return (
        f"### Instruction:\n"
        f"Classify the following email as 'Phishing' or 'Legitimate'.\n\n"
        f"### Email:\n"
        f"Subject: {subject_str}\n\n"
        f"Body:\n{body_str}\n\n"
        f"### Response:\n"
    )

def build_prompt_completion_pair(subject: str, body: str, label: int) -> Dict[str, str]:
    """
    Constructs standard {'prompt': ..., 'completion': ...} dictionary for TRL prompt-completion SFT.
    """
    label_str = "Phishing" if label == 1 else "Legitimate"
    return {
        "prompt": build_naive_prompt(subject, body),
        "completion": f" {label_str}"
    }
