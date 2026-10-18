# src/prompt_template.py — Unified Prompt Formatting for Training and Evaluation
import pandas as pd
from typing import Optional, Tuple, Dict

MAX_BODY_TOKENS = 800
MAX_SUBJECT_TOKENS = 100
MAX_SEQ_LENGTH = 1024
RESPONSE_TEMPLATE = "### Response:\n"

# Canonical completion strings — strictly NO leading space, matching prompt template ending with \n
COMPLETION_PHISHING = "Phishing"
COMPLETION_LEGITIMATE = "Legitimate"
LABEL_TO_COMPLETION = {1: COMPLETION_PHISHING, 0: COMPLETION_LEGITIMATE}

def truncate_text_tokens(text: str, max_tokens: int, tokenizer = None) -> Tuple[str, bool]:
    """
    Truncates text to a fixed token budget BEFORE prompt formatting.
    If tokenizer is provided, tokenizes and slices. Uses pre-slicing for very long strings
    to prevent quadratic tokenization slowdowns on multi-megabyte emails.
    Returns (truncated_text, was_truncated).
    """
    if not text or not isinstance(text, str):
        return "", False

    if tokenizer is not None:
        # Fast pre-cut if text is excessively long (avoids BPE quadratic blowup on 270k token text)
        if len(text) > max_tokens * 8:
            candidate = text[:max_tokens * 8]
            toks = tokenizer.encode(candidate, add_special_tokens=False)
            if len(toks) >= max_tokens:
                return tokenizer.decode(toks[:max_tokens], skip_special_tokens=True), True

        toks = tokenizer.encode(text, add_special_tokens=False)
        if len(toks) > max_tokens:
            return tokenizer.decode(toks[:max_tokens], skip_special_tokens=True), True
        return text, False
    else:
        words = text.split()
        max_words = int(max_tokens * 0.75)
        if len(words) > max_words:
            return " ".join(words[:max_words]), True
        return text, False

def truncate_body_text(body: str, max_tokens: int = MAX_BODY_TOKENS, tokenizer = None) -> Tuple[str, bool]:
    return truncate_text_tokens(body, max_tokens=max_tokens, tokenizer=tokenizer)

def truncate_subject_text(subject: str, max_tokens: int = MAX_SUBJECT_TOKENS, tokenizer = None) -> Tuple[str, bool]:
    return truncate_text_tokens(subject, max_tokens=max_tokens, tokenizer=tokenizer)

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

def build_prompt_completion_pair(subject: str, body: str, label: int, tokenizer = None) -> Dict[str, str]:
    """
    Constructs standard {'prompt': ..., 'completion': ...} dictionary for TRL prompt-completion SFT.
    Truncates subject to MAX_SUBJECT_TOKENS and body to MAX_BODY_TOKENS to guarantee sequence fits within max_length.
    Completion string has NO leading space.
    """
    s_trunc, _ = truncate_subject_text(subject, tokenizer=tokenizer)
    b_trunc, _ = truncate_body_text(body, tokenizer=tokenizer)
    label_str = LABEL_TO_COMPLETION[int(label)]
    return {
        "prompt": build_naive_prompt(s_trunc, b_trunc),
        "completion": label_str
    }
