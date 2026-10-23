import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

doc = docx.Document()

# Page Setup
for section in doc.sections:
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

PRIMARY_COLOR = RGBColor(24, 43, 73)      # Deep Navy
SECONDARY_COLOR = RGBColor(180, 83, 9)    # Amber/Gold
MUTED_TEXT = RGBColor(90, 90, 90)

# Document Title
p_title = doc.add_paragraph()
r_title = p_title.add_run("Comprehensive Research Guide & Master Blueprint:\nSmall Language Models (SLMs) vs. LLMs in Phishing Detection")
r_title.bold = True
r_title.font.size = Pt(20)
r_title.font.color.rgb = PRIMARY_COLOR
p_title.paragraph_format.space_after = Pt(4)

# Subtitle / Metadata
p_sub = doc.add_paragraph()
r_sub = p_sub.add_run(
    "Author: Satvik Patel | Mentor: Prof. Amit Sir | Institution: Ahmedabad University\n"
    "Status: Milestone at End of Week 5 (Arm A Complete) | Target Horizon: Weeks 1–8\n"
    "Codebase: https://github.com/satvikp-coder/phishing-slm-research"
)
r_sub.font.size = Pt(9.5)
r_sub.font.color.rgb = MUTED_TEXT
p_sub.paragraph_format.space_after = Pt(14)

def add_heading_1(text):
    h = doc.add_heading(text, level=1)
    h.paragraph_format.space_before = Pt(16)
    h.paragraph_format.space_after = Pt(4)
    for r in h.runs:
        r.font.color.rgb = PRIMARY_COLOR
        r.bold = True
    return h

def add_heading_2(text):
    h = doc.add_heading(text, level=2)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(3)
    for r in h.runs:
        r.font.color.rgb = SECONDARY_COLOR
        r.bold = True
    return h

def add_heading_3(text):
    h = doc.add_heading(text, level=3)
    h.paragraph_format.space_before = Pt(8)
    h.paragraph_format.space_after = Pt(2)
    for r in h.runs:
        r.bold = True
        r.font.size = Pt(11)
    return h

def format_table(table, col_widths=None):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(table.rows):
        for j, cell in enumerate(row.cells):
            cell.paragraphs[0].paragraph_format.space_after = Pt(2)
            cell.paragraphs[0].paragraph_format.space_before = Pt(2)
            for r in cell.paragraphs[0].runs:
                r.font.size = Pt(9)
                if i == 0:
                    r.bold = True
                    r.font.color.rgb = RGBColor(255, 255, 255)
            # Header background
            if i == 0:
                shading = parse_xml(r'<w:shd {} w:fill="182B49"/>'.format(nsdecls('w')))
                cell._tc.get_or_add_tcPr().append(shading)
            elif i % 2 == 1:
                shading = parse_xml(r'<w:shd {} w:fill="F8F9FA"/>'.format(nsdecls('w')))
                cell._tc.get_or_add_tcPr().append(shading)

# ==============================================================================
# SECTION 1: THE CORE RESEARCH QUESTION & WHY THIS PROJECT MATTERS
# ==============================================================================
add_heading_1("1. The Core Research Question: Why Does This Project Exist?")

doc.add_paragraph(
    "Phishing remains the single most common entry point for cyberattacks worldwide, responsible for over 80% of reported security incidents. "
    "Historically, email security relied on rule-based spam filters that searched for known 'spam words' (e.g., 'lottery', 'winner', 'urgent wire transfer') "
    "or checked domain blacklists. However, the emergence of generative AI (ChatGPT, Claude, open-source LLMs) has created a paradigm shift: "
    "cybercriminals now use AI to generate grammatically flawless, highly contextual, and personalized phishing pretexts that contain zero traditional spam words."
)

doc.add_paragraph(
    "Enterprise security teams are therefore stuck between two deeply flawed options:\n"
    "• Option A: Traditional Machine Learning (TF-IDF). It is very fast and cheap, but it only memorizes surface words. When an attacker rephrases an email with polite vocabulary, it fails completely.\n"
    "• Option B: Commercial Cloud LLMs (e.g., GPT-4). While they have deep reading comprehension, sending millions of corporate emails to a cloud API is cost-prohibitive ($1,000s/month), creates severe privacy/compliance violations (GDPR/HIPAA), and introduces massive false alarm rates (blocking legitimate business emails)."
)

doc.add_paragraph(
    "Our Research Question: Can a Small Language Model (SLM) with only 1.5 billion parameters, running entirely on-premises on standard consumer hardware (an 8 GB GPU), "
    "match or exceed the detection capabilities of massive cloud LLMs while keeping false alarms under 1%?"
)

# ==============================================================================
# SECTION 2: THE JARGON-BUSTER (DICTIONARY OF ALL CONCEPTS & TERMS)
# ==============================================================================
add_heading_1("2. Jargon-Buster: Essential Concepts Explained in Simple Terms")

doc.add_paragraph(
    "To understand our experiments and methodology, below is a clear, plain-English breakdown of all key machine learning and cybersecurity terms used throughout this project:"
)

terms = [
    ("SLM (Small Language Model)", 
     "An AI language model with relatively few parameters (typically 1B to 3B parameters), such as Qwen2.5-1.5B. Unlike 70B+ giant models, an SLM is lightweight enough to run locally inside an organization on a single laptop or desktop GPU without cloud subscriptions or privacy leaks."),

    ("LLM (Large Language Model)", 
     "A massive AI model containing tens or hundreds of billions of parameters (e.g., GPT-4, Llama-3-70B, Qwen-27B). While highly capable, they require massive data-center clusters to run, cost significant money per API query, and cannot easily be deployed inside a private enterprise firewall."),

    ("Zero-Shot Inference", 
     "Testing an AI model directly 'out of the box' without giving it any specialized training on our dataset. We simply give it an email prompt and ask: 'Is this Phishing or Legitimate?'. Our Week 4 experiments proved that zero-shot models fail in cybersecurity because they are overly paranoid."),

    ("Fine-Tuning", 
     "The process of taking a general-purpose pretrained model (which knows English grammar and general facts) and training it specifically on thousands of labeled examples (our 20,681 phishing and legitimate emails) so it becomes a specialized expert detector."),

    ("LoRA (Low-Rank Adaptation)", 
     "Standard fine-tuning requires updating all 1.5 billion parameters, which takes massive memory and ruins the original model. LoRA freezes all 1.5 billion base parameters and inserts two tiny trainable mathematical matrices (rank r=16) alongside the attention layers. This reduces trainable parameters by 99% (only ~73 MB of adapter weights) while achieving the exact same accuracy."),

    ("QLoRA (Quantized LoRA)", 
     "LoRA combined with 4-bit NormalFloat (NF4) quantization. It compresses the base model weights from 16-bit numbers to 4-bit numbers in GPU memory. This allows a 1.5B model to comfortably train within 7 GB of VRAM on an 8 GB consumer GPU (like an RTX 5070 or RTX 4060) without losing mathematical precision."),

    ("Prompt-Completion Loss Masking (-100)", 
     "A vital scientific safeguard. In our training examples, each sample has an [Input Email Prompt] and a [Target Completion: 'Phishing']. If we calculate loss on the prompt, the model wastes its capacity trying to memorize the email text. By setting the prompt labels to -100 (PyTorch's ignore index), backpropagation only penalizes the model for predicting the wrong answer."),

    ("Token Budgeting (1024 Limit)", 
     "Language models do not read raw letters; they process chunks of text called 'tokens' (~0.75 words per token). Because GPUs have strict memory limits, we enforced a mathematically strict budget: maximum 100 tokens for Subject, 800 tokens for Body, and 1,024 tokens total ceiling. This ensures 0 out-of-memory crashes across all 29,544 emails."),

    ("TF-IDF (Term Frequency-Inverse Document Frequency)", 
     "A classical statistical technique that counts how often distinctive words appear in an email compared to all emails in the dataset. Words like 'urgent account suspension' get high weights, while common words like 'the' get low weights."),

    ("Logistic Regression & Linear SVM", 
     "Traditional statistical classifiers used in Week 3. Logistic Regression fits an S-shaped probability curve (sigmoid) over word frequencies; Linear SVM finds the widest geometric gap (hyperplane) separating phishing word clusters from legitimate word clusters."),

    ("Chain-of-Thought (CoT) / Reasoning Distillation (Arm B)", 
     "Instead of training the model to guess a single word ('Phishing'), we teach it to generate a step-by-step diagnostic reasoning trace first (e.g., '1. Sender domain is spoofed. 2. Urgency is artificial. 3. Hidden link points to external server. Conclusion: Phishing'). We 'distill' this thinking ability from a large teacher LLM into our small SLM."),

    ("Adversarial Attack / AI Paraphrasing", 
     "When an attacker uses an LLM to take a known phishing email and rewrite it in polite, formal corporate English, removing all obvious spam keywords while keeping the malicious link. This is the ultimate test in Week 7 to see if Arm B is truly smarter than Arm A."),

    ("False Positive Rate (FPR)", 
     "The percentage of clean, legitimate emails that the model mistakenly flags as phishing. In real enterprise environments, FPR MUST be under 1% to 2.5%, otherwise real customer inquiries and invoices get deleted!"),

    ("Recall (Threat Catch Rate)", 
     "The percentage of actual phishing attacks that the model successfully catches. High recall is critical so dangerous attacks do not reach employees' inboxes.")
]

t_terms = doc.add_table(rows=len(terms)+1, cols=2)
t_terms.rows[0].cells[0].text = "Term / Concept"
t_terms.rows[0].cells[1].text = "Plain-English Meaning & Why It Matters in Our Research"
for i, (k, v) in enumerate(terms):
    t_terms.rows[i+1].cells[0].text = k
    t_terms.rows[i+1].cells[1].text = v
format_table(t_terms)

# ==============================================================================
# SECTION 3: THE CHRONOLOGICAL 8-WEEK BLUEPRINT (DETAILED WALKTHROUGH)
# ==============================================================================
add_heading_1("3. The Chronological 8-Week Blueprint: Step-by-Step Walkthrough")

doc.add_paragraph(
    "Our research is structured as an 8-week sequential scientific investigation. Each week addresses a specific flaw discovered in the preceding week:"
)

# WEEK 1
add_heading_2("Week 1: Research Question, Literature Review & Problem Formulation (DONE ✅)")
doc.add_paragraph(
    "• What We Did: We audited over 25 published peer-reviewed research papers on LLM cybersecurity and phishing detection. "
    "We documented how current commercial filters fail against LLM-generated phishing and defined our formal hypothesis H1.\n"
    "• Why We Did It: Without a precise hypothesis, research becomes aimless trial-and-error. We needed to guarantee that our study answers a genuine academic question: "
    "Does reasoning distillation (Arm B) impart genuine robustness against AI rephrasing compared to direct label fine-tuning (Arm A)?\n"
    "• Key Deliverables: Literature notes (`literature-notes.md`), research proposal, and the 8-week experimental protocol approved by Prof. Amit."
)

# WEEK 2
add_heading_2("Week 2: Data Acquisition & The Sealed 'Vault' Split (DONE ✅)")
doc.add_paragraph(
    "• What We Did: We acquired and standardized 29,544 raw emails from 7 respected benchmark corpora (Enron, SpamAssassin, CEAS-08, Nazario, LingSpam, Nigerian Fraud, TREC-07). "
    "Using a fixed random seed (42), we partitioned the data into three immutable splits: Train (20,681 emails, 70%), Validation (2,954 emails, 10%), and Test (5,909 emails, 20%).\n"
    "• Why We Did It (The Scientific Safeguard): In data science, 'data leakage' (letting a model see test emails during development) completely invalidates research results. "
    "To ensure publication-grade integrity, we placed `phishing_test.csv` in a sealed digital vault with strict SHA-256 cryptographic verification. It will NEVER be touched until final Week 8 evaluation.\n"
    "• Key Deliverables: Verified dataset files with SHA-256 hashes (`phishing_train.csv`, `phishing_val.csv`, and locked `phishing_test.csv`)."
)

# WEEK 3
add_heading_2("Week 3: Classical Machine Learning Baselines (DONE ✅)")
doc.add_paragraph(
    "• What We Did: We trained two gold-standard statistical machine learning models: TF-IDF + Logistic Regression and TF-IDF + Linear SVM (calibrated via Platt scaling) across a 20,000 unigram/bigram vocabulary.\n"
    "• Why We Did It: In academic research, you cannot claim a modern AI is useful unless you first benchmark it against fast, established classical methods. "
    "These models act as our scientific control.\n"
    "• What We Discovered: The classical models scored surprisingly high on standard emails (Logistic Regression achieved 97.36% Accuracy, 0.9687 F1, and 2.17% False Positive Rate). "
    "However, we identified their fatal flaw: because they only count surface keywords, they are completely blind to sentence structure and context."
)

# WEEK 4
add_heading_2("Week 4: Zero-Shot AI Testing & The 'Paranoia Collapse' (DONE ✅)")
doc.add_paragraph(
    "• What We Did: We evaluated off-the-shelf, untrained models on the 2,954 validation emails: Qwen2.5-1.5B (running locally) and Qwen-27B (via Groq Cloud API) using zero-shot classification prompts.\n"
    "• Why We Did It: Many companies wonder: 'Can we just prompt an existing LLM out-of-the-box instead of spending time fine-tuning?'. We needed empirical proof.\n"
    "• What We Discovered (Major Finding): Untrained AI models completely collapsed due to hyper-paranoia! "
    "Qwen-27B had a catastrophic 72.44% False Positive Rate (flagging 3 out of every 4 real emails as phishing), while Qwen2.5-1.5B had a 33.47% False Positive Rate (58.11% F1). "
    "This mathematically proved that out-of-the-box prompting is completely unusable in enterprise security, establishing the absolute necessity for specialized fine-tuning."
)

# WEEK 5
add_heading_2("Week 5: Arm A (Naive QLoRA) Fine-Tuning on RTX 5070 GPU (DONE ✅)")
doc.add_paragraph(
    "• What We Did: We fine-tuned Qwen2.5-1.5B for 3 full epochs on our 20,681 training emails using 16-bit LoRA on an NVIDIA GeForce RTX 5070 GPU. "
    "We verified 100% prompt loss masking and evaluated the model against all 2,954 validation emails.\n"
    "• Why We Did It: Arm A is the first of our two experimental arms. It teaches the model direct classification ('Phishing' vs 'Legitimate') under identical conditions.\n"
    "• The Milestone Results:\n"
    "  - Accuracy: 99.15% (2,929 out of 2,954 correct)\n"
    "  - F1-Score: 0.9900 (surpassing classical baselines)\n"
    "  - False Positive Rate: 0.88% (a massive 60% reduction in false alarms compared to classical models!)\n"
    "  - Parse Failure Rate: 0.00% (100% compliant outputs)\n"
    "  - All checkpoints, prediction logs, and code were synchronized and committed to GitHub."
)

# WEEK 6
add_heading_2("Week 6: Arm B (Chain-of-Thought Reasoning Distillation) (UP NEXT ⏳)")
doc.add_paragraph(
    "• The Goal: Rather than outputting just a label, train the SLM to generate a diagnostic explanation before its verdict (analyzing sender identity, urgency tricks, domain mismatches, and call-to-action).\n"
    "• Step 1 (Distillation): Use an advanced teacher model (via high-speed Groq API) to generate high-quality reasoning rationales for our training emails, producing `phishing_train_cot.csv`.\n"
    "• Step 2 (GPU Training): Fine-tune Qwen2.5-1.5B on these reasoning traces on the RTX 5070 GPU (~5.5 to 6.5 hours of training).\n"
    "• The Deliverable: A lightweight SLM capable of full explainable security reporting: `[Reasoning Trace] -> Verdict`."
)

# WEEK 7
add_heading_2("Week 7: The Adversarial Stress Test (AI vs. AI Battle) (UPCOMING ⏳)")
doc.add_paragraph(
    "• The Goal: Test the central hypothesis of our entire research project: Is Arm B more robust against AI-generated attacks than Arm A?\n"
    "• The Experiment: We will take real phishing emails and use ChatGPT to rewrite them into polished, formal business communications that contain zero obvious spam words (simulating modern advanced spear-phishing).\n"
    "• The Scientific Showdown:\n"
    "  - Does Arm A get fooled because it only memorized word associations?\n"
    "  - Does Arm B successfully detect the attack because its reasoning chain checks the actual technical intent (domain mismatch, suspicious redirection)?\n"
    "• This experiment provides the core novelty and empirical contribution for our research paper."
)

# WEEK 8
add_heading_2("Week 8: The Grand Finale — Test Set Unlock & Comprehensive Paper (UPCOMING ⏳)")
doc.add_paragraph(
    "• The Goal: Unseal the digital vault (`phishing_test.csv` — 5,909 emails) for the official, final benchmark.\n"
    "• What We Will Measure:\n"
    "  1. Final Generalization: Unbiased F1, Accuracy, Precision, Recall, and AUROC across all models.\n"
    "  2. Latency & Throughput: Milliseconds per email on local GPU vs cloud API.\n"
    "  3. Cost Analysis: Dollar cost per 10,000 emails ($0.00 local SLM vs cloud LLM pricing).\n"
    "  4. Hardware Footprint: Peak VRAM, parameters, and energy consumption.\n"
    "• Final Deliverable: The complete research thesis paper and presentation slides for Prof. Amit Sir."
)

# ==============================================================================
# SECTION 4: MASTER COMPARISON SCORECARD
# ==============================================================================
add_heading_1("4. Master Benchmark Scorecard: What the Numbers Tell Us So Far")

scorecard_rows = [
    ("Qwen-27B (Groq Cloud)", "Zero-Shot LLM", "32.54%", "28.64%", "39.26%", "0.3312", "72.44% (Fatal)", "38.1 ms", "Unusable due to 72% false alarm rate"),
    ("Qwen2.5-1.5B-Instruct", "Zero-Shot SLM", "63.64%", "56.61%", "59.70%", "0.5811", "33.47% (High)", "432.5 ms", "Untrained model is overly paranoid"),
    ("TF-IDF + Linear SVM", "Classical ML", "97.02%", "96.77%", "96.15%", "0.9646", "2.34% (Good)", "0.51 ms", "High accuracy, but brittle to AI paraphrasing"),
    ("TF-IDF + Logistic Reg.", "Classical ML", "97.36%", "97.03%", "96.71%", "0.9687", "2.17% (Good)", "0.43 ms", "Strong statistical baseline; zero reasoning"),
    ("Qwen2.5-1.5B (Arm A)", "Fine-Tuned SLM", "99.15%", "98.80%", "99.20%", "0.9900", "0.88% (Superior)", "167.6 ms", "Best overall; 60% fewer false alarms than TF-IDF"),
    ("Qwen2.5-1.5B (Arm B)", "CoT Distilled SLM", "Pending", "Pending", "Pending", "Pending", "Pending", "~350 ms", "Scheduled for Week 6 execution")
]

t_score = doc.add_table(rows=len(scorecard_rows)+1, cols=9)
headers_score = ["Model / System", "Type", "Accuracy", "Precision", "Recall", "F1 Score", "False Positives (FPR)", "Latency", "Current Status / Verdict"]
for j, h in enumerate(headers_score):
    t_score.rows[0].cells[j].text = h
for i, row in enumerate(scorecard_rows):
    for j, val in enumerate(row):
        t_score.rows[i+1].cells[j].text = val
format_table(t_score)

# ==============================================================================
# SECTION 5: SUMMARY & NEXT IMMEDIATE ACTIONS
# ==============================================================================
add_heading_1("5. Summary & Next Immediate Actions")

doc.add_paragraph(
    "1. Where We Stand Today: The first major research phase is a total success. We proved that a 1.5B Small Language Model, "
    "when fine-tuned with QLoRA, achieves 99.15% accuracy and outperforms classical machine learning baselines while reducing false positives to 0.88%.\n\n"
    "2. Next Immediate Step: Begin Week 6 (Arm B Data Distillation). We will generate the Chain-of-Thought reasoning dataset, verify the reasoning traces, "
    "and prepare the training script for execution on the RTX 5070 GPU.\n\n"
    "3. Deliverables for Prof. Amit: The formal update email, the detailed validation report (`Arm_A_Fine_Tuning_Validation_Report.docx`), "
    "and this Master Blueprint document (`Project_Master_Blueprint_And_Status.docx`) provide complete, transparent documentation of all experimental progress."
)

# Save
docx_out = r"c:\Users\satvi\Desktop\Amit Sir\Project_Master_Blueprint_And_Status.docx"
doc.save(docx_out)
print("Successfully generated Word document:", docx_out)
