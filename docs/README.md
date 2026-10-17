# Documentation Directory (`docs/`)

This directory contains research context, literature analyses, verification reports, and handover documentation.

---

## Directory Organization

```
docs/
├── PROGRESS_AND_RESULTS_EVIDENCE.md # Master academic evidence report with comparative inferences
├── handover/                        # Structured handover & context packages (00 to 10)
│   ├── 00_START_HERE.md             # Onboarding map for researchers and agents
│   ├── 01_RESEARCH_MASTER_CONTEXT.md# Research questions, hypotheses, and scope
│   ├── 02_RESEARCH_HISTORY_AND_DECISIONS.md # Architectural decisions and rationale
│   ├── 03_DATASETS_AND_PREPROCESSING.md # Data curation details
│   ├── 04_CODEBASE_AND_TECHNICAL_ARCHITECTURE.md # Codebase structure
│   ├── 05_EXPERIMENTS_AND_RESULTS.md # Experimental results log
│   ├── 06_ERRORS_AND_UNRESOLVED_ISSUES.md # Solved and active hazards
│   ├── 07_CURRENT_STATE_AND_NEXT_STEPS.md # Active roadmap
│   ├── 08_CODEX_OPERATING_INSTRUCTIONS.md # Operating rules and test set lock protocol
│   ├── 09_CONTEXT_GAPS_AND_VERIFICATION.md # Checklist for verification
│   ├── 10_CODEX_BOOTSTRAP_PROMPT.md # Prompt for initializing new agent sessions
│   └── VERIFICATION_REPORT.md       # Formal data verification & anti-leakage audit
└── README.md                        # This guide
```

---

## Key Documents to Consult

1. **For Supervisor Meetings / Audit:**
   - [`PROGRESS_AND_RESULTS_EVIDENCE.md`](file:///docs/PROGRESS_AND_RESULTS_EVIDENCE.md): The full evidence document summarizing verification, baselines, and academic inferences.
2. **For Dataset Integrity:**
   - [`handover/VERIFICATION_REPORT.md`](file:///docs/handover/VERIFICATION_REPORT.md): Step 1 verification report with exact row counts and SHA-256 signatures.
3. **For Research Rules:**
   - [`handover/08_CODEX_OPERATING_INSTRUCTIONS.md`](file:///docs/handover/08_CODEX_OPERATING_INSTRUCTIONS.md): Anti-leakage constraints and fair-comparison rules.
