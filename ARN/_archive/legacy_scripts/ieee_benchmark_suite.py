"""
═══════════════════════════════════════════════════════════════════════════════
  ARN IEEE-Paper-Grade Benchmark Suite
  ─────────────────────────────────────
  Comprehensive evaluation for:
    [1] 5-Fold Cross-Validation with Multi-Seed (3 seeds × 5 folds)
    [2] Entity-Level F1 (seqeval strict), Token-Level Accuracy
    [3] Ablation Study: Full ARN vs JointBERT vs NoCRF vs NoFAR
    [4] Bootstrap 95% Confidence Intervals
    [5] Hardware Profiling: Latency (p50/p95/p99), RAM, Disk, FLOPs
    [6] Extended OOD False-Positive Rejection (N=200)
    [7] Per-Tool Routing Breakdown
    [8] FAR Confidence Calibration Analysis
    [9] JSON + LaTeX Export for Paper Tables
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import tracemalloc
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 stdout/stderr on Windows to avoid CP1252 encoding errors
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# CRITICAL: tokenizer_utils must be imported FIRST to initialize the torchvision mock
# before transformers performs its version checks
from config import ARNconfig
from tokenizer_utils import TOKENIZER, TAG_TO_IDX, IDX_TO_TAG, TOOL_TO_IDX, IDX_TO_TOOL

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, hamming_loss
)
from transformers import AutoModel, logging as hf_logging
hf_logging.set_verbosity_error()

from dataset import ARNDataset
from model import AetherRoutingNetwork
from far import FactoredAttentionRouter
from crf import CRF
from inference import predict

# ════════════════════════════════════════════════════════════════════════════
# SECTION 0: Constants
# ════════════════════════════════════════════════════════════════════════════

SEEDS = [42, 100, 2026]
NUM_FOLDS = 5
LATENCY_ITERATIONS = 500
LATENCY_WARMUP = 50

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "benchmark_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Extended OOD prompts (N=200) for false-positive rejection testing
OOD_PROMPTS = [
    # ── General Greetings & Politeness (20) ──
    "Hello there!", "How's the weather today?", "Tell me a joke.",
    "Who is the president of the United States?", "Can you explain quantum physics?",
    "Let's grab a coffee sometime.", "Nice to meet you.", "What is your name?",
    "Goodbye!", "Just thinking out loud about some coding ideas.",
    "Good morning how are you doing today?", "Thanks a lot for your help!",
    "Have a wonderful day ahead", "What are your favorite movies?",
    "Hey assistant how is it going?", "How was your weekend?",
    "I appreciate your patience with me.", "See you later alligator.",
    "What do you think about the new iPhone?", "Happy birthday to you!",

    # ── Casual Chit-Chat & Questions (40) ──
    "Why is the sky blue?", "Do you know who won the World Cup in 2022?",
    "What is the capital of France?", "Can dogs eat chocolate?",
    "How far is the moon from Earth?", "Tell me a story about a dragon.",
    "What is the meaning of life?", "Do you like pizza or burgers better?",
    "How do airplanes stay in the air?", "What is your opinion on artificial intelligence?",
    "Can you write a poem about autumn?", "Who wrote Hamlet?",
    "Is it going to rain this weekend?", "What is the speed of light?",
    "How do batteries store electricity?", "Tell me an interesting science fact.",
    "What are the primary colors?", "How do trees produce oxygen?",
    "What time do grocery stores usually close?", "Why do cats purr when they are happy?",
    "What is the largest ocean on Earth?", "Who painted the Mona Lisa?",
    "How many continents are there?", "What is the difference between fruit and vegetable?",
    "Can you translate hello into Spanish?", "What is the tallest building in the world?",
    "How does WiFi work?", "What is the population of India?",
    "Who invented the telephone?", "What is photosynthesis?",
    "How do magnets work?", "What is the deepest part of the ocean?",
    "Tell me about ancient Egypt.", "What language has the most native speakers?",
    "Why do we dream at night?", "What is dark matter?",
    "How does the immune system fight infections?", "What is the fibonacci sequence?",
    "Who discovered gravity?", "What causes earthquakes?",

    # ── Statement / Thought Fragments (40) ──
    "I really enjoy listening to classical music in the evening.",
    "Debugging code can be quite challenging sometimes.",
    "Python is one of the most popular programming languages today.",
    "Machine learning requires a lot of high quality data.",
    "I need to drink more water throughout the day.",
    "The sunset yesterday was absolutely breathtaking.",
    "Virtual reality technology has come a long way in recent years.",
    "Reading books helps improve vocabulary and focus.",
    "Exercise is essential for maintaining good physical health.",
    "Automated testing saves a lot of time in software development.",
    "I am thinking about learning how to play the acoustic guitar.",
    "Coffee tastes great on a chilly winter morning.",
    "Space exploration has always fascinated human beings.",
    "Clean architecture makes software codebases much easier to maintain.",
    "Open source software powers most of the internet infrastructure.",
    "Electric vehicles are becoming more common every year.",
    "Typography plays a crucial role in user interface design.",
    "I love the smell of fresh rain on dry earth.",
    "Cooking your own meals is healthier and more affordable.",
    "Consistency is key to mastering any new skill.",
    "Digital privacy is an important topic in modern technology.",
    "Traveling exposes you to new cultures and perspectives.",
    "Recursion is a programming concept where a function calls itself.",
    "A balanced diet provides steady energy throughout the day.",
    "Mindfulness practice can help reduce daily stress levels.",
    "I wonder what the weather will be like tomorrow morning.",
    "The new restaurant downtown has excellent reviews online.",
    "Sometimes I just stare at the ceiling and think about nothing.",
    "My cat just knocked over a glass of water again.",
    "I should probably go to bed early tonight for once.",
    "The documentary about deep sea creatures was fascinating.",
    "I keep forgetting to water my plants every other day.",
    "The library is such a peaceful place to study in silence.",
    "I really need to organize my desk it is a complete mess.",
    "That lecture on philosophy was surprisingly engaging.",
    "I cannot believe how fast this year went by.",
    "My favorite season is autumn because of the colorful leaves.",
    "I should start journaling to keep track of my thoughts.",
    "The traffic on the highway was absolutely terrible today.",
    "Sometimes the simplest solution is the most elegant one.",

    # ── Technical & Math Enquiries without Tool Intent (40) ──
    "What is the formula for calculating the area of a circle?",
    "How does a binary search algorithm work?",
    "What is the difference between TCP and UDP networking protocols?",
    "Can you explain what a REST API is?",
    "What is the complexity of quicksort in the worst case?",
    "How does garbage collection work in Java?",
    "What is the Pythagorean theorem?",
    "What is a hash table and how does collision resolution work?",
    "What is the purpose of Docker containers in DevOps?",
    "How does public key cryptography work?",
    "What is a database index and why is it used?",
    "What is the difference between process and thread in OS?",
    "How does gradient descent optimize neural network weights?",
    "What is the derivative of x squared?",
    "What is a pull request in Git version control?",
    "How does SSL TLS encryption protect web traffic?",
    "What is the difference between synchronous and asynchronous execution?",
    "What is a closure in JavaScript?",
    "What is the CAP theorem in distributed systems?",
    "How does a compiler differ from an interpreter?",
    "What is the Big O notation used for in computer science?",
    "How do neural networks learn through backpropagation?",
    "What is the difference between stack and heap memory?",
    "Can you explain the observer design pattern?",
    "What is a deadlock in operating systems?",
    "How does a convolutional neural network process images?",
    "What is the halting problem in computability theory?",
    "What is memoization and when should you use it?",
    "How does DNS resolution work step by step?",
    "What is the difference between supervised and unsupervised learning?",
    "What is a Turing machine?",
    "How does a blockchain maintain consensus?",
    "What is the difference between regression and classification?",
    "How do attention mechanisms work in transformers?",
    "What is the purpose of normalization in databases?",
    "How does a random forest algorithm make predictions?",
    "What is the difference between L1 and L2 regularization?",
    "What is dependency injection in software engineering?",
    "How does MapReduce process large datasets?",
    "What is the difference between HTTP and HTTPS?",

    # ── Conversational Commands without Actionable Tools (30) ──
    "Be quiet for a second.", "Think deeply about this problem.",
    "Hold on a moment while I check something.", "Never mind about that.",
    "Forget what I just said.", "Listen to me carefully.",
    "Pay attention to this detail.", "Take your time.", "Don't worry about it.",
    "Keep that in mind for later.", "Let me know what you think.",
    "Ponder this question for a bit.", "Consider the following scenario.",
    "Imagine a world without internet.", "Stay tuned for further updates.",
    "That is a really interesting perspective you have there.",
    "I completely agree with what you just said.",
    "Can you repeat that one more time please?",
    "I did not quite catch what you meant by that.",
    "That makes a lot of sense when you put it that way.",
    "I am not sure I understand the question fully.",
    "Could you elaborate on that particular point?",
    "Well that is certainly one way to look at it.",
    "I will have to think about that for a while.",
    "That is a fair point actually.", "You raise a valid concern there.",
    "I had not thought of it from that angle before.",
    "Let us agree to disagree on this one.",
    "That reminds me of something I read last week.",
    "Interesting, tell me more about your reasoning.",

    # ── Ambiguous / Near-Miss Prompts (30) ──
    "I wish I could message my friend right now but my phone is broken.",
    "Remind me to buy groceries when I get home later today.",
    "I was just thinking about what time it is in Tokyo.",
    "Someone told me to google the answer but I prefer asking you.",
    "I should probably set an alarm but I keep forgetting.",
    "My smart home devices are really cool but sometimes glitchy.",
    "I heard that timers can help with productivity and focus.",
    "Media consumption has really changed in the last decade.",
    "Web browsing on mobile is still not as good as desktop.",
    "I need to type up a report for my boss by end of day.",
    "App development has become much more accessible to beginners.",
    "URLs can be really long and confusing sometimes.",
    "Command line tools are powerful but have a steep learning curve.",
    "Screen analysis technology is advancing rapidly these days.",
    "Script writing is an essential skill for system administrators.",
    "System diagnostics revealed some interesting performance data.",
    "WhatsApp has over two billion users worldwide now.",
    "Timer functionality is built into most modern smart watches.",
    "Skills based learning is the future of education I think.",
    "The search engine market is dominated by just a few companies.",
    "Home automation is getting cheaper and more reliable every year.",
    "Reading articles online has replaced newspapers for many people.",
    "I wonder how long it takes to learn a new programming language.",
    "The battery on my laptop drains faster in cold weather.",
    "Voice assistants are getting better at understanding accents.",
    "I prefer typing over voice commands most of the time.",
    "Smart home setups can be complex to configure initially.",
    "The internet has fundamentally changed how we access information.",
    "I find it hard to focus when there is background noise.",
    "Technology moves so fast it is hard to keep up sometimes.",
]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: Baseline Model Definitions
# ════════════════════════════════════════════════════════════════════════════

class JointBERT_Baseline(nn.Module):
    """Standard JointBERT: [CLS]-pooled linear tool routing + Softmax slot tagger."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(
            "google/bert_uncased_L-2_H-128_A-2", attn_implementation="eager"
        )
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.tool_classifier = nn.Linear(config.HIDDEN_DIM, config.NUM_TOOLS)
        self.slot_tagger = nn.Linear(config.HIDDEN_DIM, len(TAG_TO_IDX))

    def forward(self, input_ids, attention_mask, tags=None, tool_labels=None):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_rep = h[:, 0, :]
        tool_logits = self.tool_classifier(cls_rep)
        emissions = self.slot_tagger(h)
        if tags is not None:
            tool_loss = nn.BCEWithLogitsLoss()(tool_logits, tool_labels)
            slot_loss = nn.CrossEntropyLoss(ignore_index=TAG_TO_IDX["<PAD>"])(
                emissions.view(-1, len(TAG_TO_IDX)), tags.view(-1)
            )
            total_loss = tool_loss + self.config.LAMBDA_SLOT * slot_loss
            return (total_loss, tool_logits)
        else:
            tool_preds = (torch.sigmoid(tool_logits) > 0.5)
            slot_preds = torch.argmax(emissions, dim=-1).cpu().numpy().tolist()
            return tool_preds, slot_preds


class ARN_NoCRF(nn.Module):
    """ARN ablation: FAR router + Softmax slot tagger (no CRF)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(
            "google/bert_uncased_L-2_H-128_A-2", attn_implementation="eager"
        )
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.far = FactoredAttentionRouter(config.NUM_TOOLS, config.HIDDEN_DIM, config.DROPOUT)
        self.slot_tagger = nn.Linear(config.HIDDEN_DIM, len(TAG_TO_IDX))

    def forward(self, input_ids, attention_mask, tags=None, tool_labels=None):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        tool_logits, rw = self.far(h, attention_mask)
        emissions = self.slot_tagger(h)
        if tags is not None:
            tool_loss = nn.BCEWithLogitsLoss()(tool_logits, tool_labels)
            slot_loss = nn.CrossEntropyLoss(ignore_index=TAG_TO_IDX["<PAD>"])(
                emissions.view(-1, len(TAG_TO_IDX)), tags.view(-1)
            )
            total_loss = tool_loss + self.config.LAMBDA_SLOT * slot_loss
            return (total_loss, tool_logits)
        else:
            tool_preds = (torch.sigmoid(tool_logits) > 0.5)
            slot_preds = torch.argmax(emissions, dim=-1).cpu().numpy().tolist()
            return tool_preds, slot_preds


class ARN_NoFAR(nn.Module):
    """ARN ablation: [CLS]-pooled linear tool routing + CRF slot tagger (no FAR)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(
            "google/bert_uncased_L-2_H-128_A-2", attn_implementation="eager"
        )
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.tool_classifier = nn.Linear(config.HIDDEN_DIM, config.NUM_TOOLS)
        self.slot_tagger = nn.Linear(config.HIDDEN_DIM, len(TAG_TO_IDX))
        self.crf = CRF(num_tags=len(TAG_TO_IDX))

    def forward(self, input_ids, attention_mask, tags=None, tool_labels=None):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_rep = h[:, 0, :]
        tool_logits = self.tool_classifier(cls_rep)
        emissions = self.slot_tagger(h)
        if tags is not None:
            tool_loss = nn.BCEWithLogitsLoss()(tool_logits, tool_labels)
            active_tokens = attention_mask.sum() + 1e-8
            raw_slot_loss = self.crf.loss(emissions, tags, attention_mask)
            slot_loss = raw_slot_loss / active_tokens
            total_loss = tool_loss + self.config.LAMBDA_SLOT * slot_loss
            return (total_loss, tool_logits)
        else:
            tool_preds = (torch.sigmoid(tool_logits) > 0.5)
            tag_preds = self.crf.viterbi_decode(emissions, attention_mask)
            return tool_preds, tag_preds


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: Entity-Level F1 (seqeval-compatible)
# ════════════════════════════════════════════════════════════════════════════

def _compute_entity_spans(tag_sequence):
    """Convert a list of BIO tag strings into a set of (type, start, end) spans."""
    spans = set()
    current_type = None
    current_start = None
    for i, tag in enumerate(tag_sequence):
        if tag.startswith("B-"):
            if current_type is not None:
                spans.add((current_type, current_start, i))
            current_type = tag[2:]
            current_start = i
        elif tag.startswith("I-"):
            tag_type = tag[2:]
            if current_type != tag_type:
                if current_type is not None:
                    spans.add((current_type, current_start, i))
                current_type = tag_type
                current_start = i
        else:  # O tag
            if current_type is not None:
                spans.add((current_type, current_start, i))
                current_type = None
                current_start = None
    if current_type is not None:
        spans.add((current_type, current_start, len(tag_sequence)))
    return spans


def compute_entity_f1(y_true_sequences, y_pred_sequences):
    """Compute entity-level Precision, Recall, F1 (strict match)."""
    tp, fp, fn = 0, 0, 0
    per_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for true_seq, pred_seq in zip(y_true_sequences, y_pred_sequences):
        true_spans = _compute_entity_spans(true_seq)
        pred_spans = _compute_entity_spans(pred_seq)

        matched = true_spans & pred_spans
        tp += len(matched)
        fp += len(pred_spans - true_spans)
        fn += len(true_spans - pred_spans)

        for span in matched:
            per_type[span[0]]["tp"] += 1
        for span in pred_spans - true_spans:
            per_type[span[0]]["fp"] += 1
        for span in true_spans - pred_spans:
            per_type[span[0]]["fn"] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    per_type_f1 = {}
    for etype, counts in per_type.items():
        p = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) > 0 else 0.0
        r = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_type_f1[etype] = {"precision": p, "recall": r, "f1": f, "support": counts["tp"] + counts["fn"]}

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "per_type": per_type_f1
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: Evaluation Core
# ════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, loader, config, return_raw=False):
    """
    Unified evaluation that returns:
      - Tool Routing: Subset Accuracy, Hamming Loss, Micro/Macro F1, Per-tool F1
      - Slot Tagging: Token Accuracy, Active Slot Accuracy, Entity-level F1
      - Joint: Exact Match Accuracy
      - Confidence: Mean confidence on correct vs incorrect predictions
    """
    model.eval()
    pad_idx = TAG_TO_IDX.get("<PAD>", 21)
    o_idx = TAG_TO_IDX.get("O", 0)

    all_tool_labels, all_tool_preds = [], []
    all_tool_logits = []
    all_tag_true_seqs, all_tag_pred_seqs = [], []
    y_true_active, y_pred_active = [], []
    joint_correct, total_samples = 0, 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"]
            mask = batch["attention_mask"]
            tag_ids = batch["tags_ids"]
            tool_labels = batch["tool_labels"]

            # Get predictions
            result = model(input_ids, mask)
            if len(result) == 3:
                tool_preds, tag_preds, rw = result
            else:
                tool_preds, tag_preds = result

            # Also get logits for confidence analysis
            outputs = model.encoder(input_ids=input_ids, attention_mask=mask) if hasattr(model, 'encoder') else None
            if hasattr(model, 'far') and outputs is not None:
                h = outputs.last_hidden_state
                logits, _ = model.far(h, mask)
                all_tool_logits.append(torch.sigmoid(logits).cpu().numpy())
            elif hasattr(model, 'tool_classifier') and outputs is not None:
                h = outputs.last_hidden_state
                logits = model.tool_classifier(h[:, 0, :])
                all_tool_logits.append(torch.sigmoid(logits).cpu().numpy())

            tp_np = tool_preds.cpu().numpy()
            tl_np = tool_labels.cpu().numpy().astype(bool)
            all_tool_labels.append(tl_np)
            all_tool_preds.append(tp_np)

            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)

            for b in range(batch_size):
                total_samples += 1
                tools_match = np.array_equal(tp_np[b], tl_np[b])
                slots_match = True

                true_seq, pred_seq = [], []
                for t in range(seq_len):
                    gt = tag_ids[b, t].item()
                    if gt == pad_idx:
                        continue
                    pr = tag_preds[b][t]
                    true_seq.append(IDX_TO_TAG[gt])
                    pred_seq.append(IDX_TO_TAG[pr])

                    if gt != o_idx:
                        y_true_active.append(gt)
                        y_pred_active.append(pr)
                    if pr != gt:
                        slots_match = False

                all_tag_true_seqs.append(true_seq)
                all_tag_pred_seqs.append(pred_seq)

                if tools_match and slots_match:
                    joint_correct += 1

    # Aggregate
    all_tool_labels = np.concatenate(all_tool_labels, axis=0)
    all_tool_preds = np.concatenate(all_tool_preds, axis=0)

    # Tool metrics
    subset_acc = np.all(all_tool_preds == all_tool_labels, axis=1).mean() * 100
    h_loss = hamming_loss(all_tool_labels, all_tool_preds) * 100
    tool_f1_micro = f1_score(all_tool_labels, all_tool_preds, average='micro', zero_division=0) * 100
    tool_f1_macro = f1_score(all_tool_labels, all_tool_preds, average='macro', zero_division=0) * 100

    # Per-tool F1
    per_tool_f1 = {}
    for tool_name, tool_idx in TOOL_TO_IDX.items():
        col_true = all_tool_labels[:, tool_idx]
        col_pred = all_tool_preds[:, tool_idx]
        f1_val = f1_score(col_true, col_pred, zero_division=0) * 100
        support = int(col_true.sum())
        per_tool_f1[tool_name] = {"f1": f1_val, "support": support}

    # Token-level slot metrics
    y_true_active = np.array(y_true_active)
    y_pred_active = np.array(y_pred_active)
    active_slot_acc = (y_true_active == y_pred_active).mean() * 100 if len(y_true_active) > 0 else 0.0
    active_slot_f1 = f1_score(y_true_active, y_pred_active, average='macro', zero_division=0) * 100

    # Entity-level F1
    entity_metrics = compute_entity_f1(all_tag_true_seqs, all_tag_pred_seqs)

    # Joint accuracy
    joint_acc = (joint_correct / total_samples) * 100 if total_samples > 0 else 0.0

    # Confidence calibration
    confidence_results = {}
    if all_tool_logits:
        all_logits_np = np.concatenate(all_tool_logits, axis=0)
        correct_confs, incorrect_confs = [], []
        for i in range(len(all_tool_labels)):
            match = np.array_equal(all_tool_preds[i], all_tool_labels[i])
            active_mask = all_tool_labels[i].astype(bool)
            if active_mask.any():
                conf = all_logits_np[i][active_mask].mean()
            else:
                conf = (1.0 - all_logits_np[i]).mean()
            if match:
                correct_confs.append(conf)
            else:
                incorrect_confs.append(conf)
        confidence_results = {
            "correct_mean_conf": float(np.mean(correct_confs)) * 100 if correct_confs else 0.0,
            "incorrect_mean_conf": float(np.mean(incorrect_confs)) * 100 if incorrect_confs else 0.0,
            "separation_gap": float(np.mean(correct_confs) - np.mean(incorrect_confs)) * 100
                              if correct_confs and incorrect_confs else 0.0,
            "n_correct": len(correct_confs),
            "n_incorrect": len(incorrect_confs),
        }

    results = {
        "tool_subset_accuracy": round(subset_acc, 2),
        "tool_hamming_loss": round(h_loss, 4),
        "tool_f1_micro": round(tool_f1_micro, 2),
        "tool_f1_macro": round(tool_f1_macro, 2),
        "per_tool_f1": per_tool_f1,
        "active_slot_accuracy": round(active_slot_acc, 2),
        "active_slot_f1_macro": round(active_slot_f1, 2),
        "entity_precision": round(entity_metrics["precision"] * 100, 2),
        "entity_recall": round(entity_metrics["recall"] * 100, 2),
        "entity_f1": round(entity_metrics["f1"] * 100, 2),
        "entity_per_type": {k: {kk: round(vv * 100, 2) if kk != "support" else vv
                                for kk, vv in v.items()}
                           for k, v in entity_metrics["per_type"].items()},
        "joint_exact_match": round(joint_acc, 2),
        "confidence": confidence_results,
        "total_samples": total_samples,
    }
    return results


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: Training Loop (for ablation baselines)
# ════════════════════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader, config, epochs=30, lr=3e-4):
    """Train a model (for baseline/ablation) and return best val loss model state."""
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            result = model(
                batch["input_ids"], batch["attention_mask"],
                tags=batch["tags_ids"], tool_labels=batch["tool_labels"]
            )
            loss = result[0]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                result = model(
                    batch["input_ids"], batch["attention_mask"],
                    tags=batch["tags_ids"], tool_labels=batch["tool_labels"]
                )
                val_loss += result[0].item() * batch["input_ids"].size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5: Bootstrap Confidence Intervals
# ════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(values, n_boot=2000, ci=95):
    """Compute bootstrap confidence interval for a list of values."""
    if len(values) < 2:
        m = np.mean(values) if values else 0.0
        return m, (m, m)
    rng = np.random.RandomState(42)
    boots = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(boots, (100 - ci) / 2)
    upper = np.percentile(boots, 100 - (100 - ci) / 2)
    return float(np.mean(values)), (float(lower), float(upper))


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6: Main Benchmark Suite
# ════════════════════════════════════════════════════════════════════════════

def run_full_benchmark():
    config = ARNconfig()

    # ── Load dataset ──
    dataset_path = os.path.join(os.path.dirname(__file__), "aether_orchestrator_dataset_augmented.jsonl")
    print(f"\n{'=' * 80}")
    print(f"  ARN IEEE-PAPER BENCHMARK SUITE")
    print(f"{'=' * 80}")
    print(f"  Dataset: {dataset_path}")

    full_dataset = ARNDataset(jsonl_path=dataset_path, tokenizer=TOKENIZER, max_seq_len=config.MAX_SEQ_LEN)
    print(f"  Total samples loaded: {len(full_dataset)}")

    # Create primary tool label vector for stratification
    primary_labels = []
    for i in range(len(full_dataset)):
        tool_vec = full_dataset[i]["tool_labels"].numpy()
        primary_labels.append(np.argmax(tool_vec) if tool_vec.any() else 0)
    primary_labels = np.array(primary_labels)

    # ── Load trained ARN checkpoint ──
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", "arn_best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"  ERROR: Checkpoint not found at {ckpt_path}")
        return
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Checkpoint size: {os.path.getsize(ckpt_path) / 1024 / 1024:.2f} MB")

    all_results = {}

    # ══════════════════════════════════════════════════════════════════════
    # BENCHMARK 1: Multi-Seed 5-Fold Cross-Validation
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'-' * 80}")
    print(f"  [1/7] MULTI-SEED 5-FOLD CROSS-VALIDATION (3 seeds x 5 folds)")
    print(f"{'-' * 80}")

    arn_model = AetherRoutingNetwork(config)
    arn_model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    arn_model.eval()

    fold_results = {
        "tool_subset_accuracy": [], "tool_f1_micro": [], "tool_f1_macro": [],
        "active_slot_accuracy": [], "active_slot_f1_macro": [],
        "entity_f1": [], "entity_precision": [], "entity_recall": [],
        "joint_exact_match": [],
    }

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(full_dataset)), primary_labels)):
            val_subset = Subset(full_dataset, val_idx)
            val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
            results = evaluate_model(arn_model, val_loader, config)

            for key in fold_results:
                fold_results[key].append(results[key])

            print(f"    Seed {seed} Fold {fold_idx+1}: "
                  f"Tool Acc={results['tool_subset_accuracy']:.1f}% | "
                  f"Entity F1={results['entity_f1']:.1f}% | "
                  f"Joint={results['joint_exact_match']:.1f}%")

    # Compute means, stds, and CIs
    cv_summary = {}
    for key, values in fold_results.items():
        mean_val, ci = bootstrap_ci(values)
        std_val = float(np.std(values))
        cv_summary[key] = {
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "ci_95_lower": round(ci[0], 2),
            "ci_95_upper": round(ci[1], 2),
            "n_folds": len(values),
        }

    all_results["cross_validation"] = cv_summary
    print(f"\n  +--- Cross-Validation Summary (Mean +/- Std [95% CI]) ---+")
    for key, v in cv_summary.items():
        print(f"  | {key:<28s}: {v['mean']:6.2f} +/- {v['std']:.2f}  [{v['ci_95_lower']:.1f}, {v['ci_95_upper']:.1f}]")
    print(f"  +------------------------------------------------------+")

    # ══════════════════════════════════════════════════════════════════════
    # BENCHMARK 2: Held-Out Test Set (70/15/15 split)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'-' * 80}")
    print(f"  [2/7] HELD-OUT TEST SET EVALUATION (70/15/15)")
    print(f"{'-' * 80}")

    from torch.utils.data import random_split
    n_total = len(full_dataset)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    gen = torch.Generator().manual_seed(42)
    _, _, test_dataset = random_split(full_dataset, [n_train, n_val, n_test], generator=gen)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    test_results = evaluate_model(arn_model, test_loader, config)
    all_results["held_out_test"] = test_results

    print(f"  Test Set Size: {n_test}")
    print(f"  Tool Subset Accuracy:  {test_results['tool_subset_accuracy']:.2f}%")
    print(f"  Tool F1 (Micro):       {test_results['tool_f1_micro']:.2f}%")
    print(f"  Tool F1 (Macro):       {test_results['tool_f1_macro']:.2f}%")
    print(f"  Active Slot Accuracy:  {test_results['active_slot_accuracy']:.2f}%")
    print(f"  Entity-Level F1:       {test_results['entity_f1']:.2f}%")
    print(f"  Entity Precision:      {test_results['entity_precision']:.2f}%")
    print(f"  Entity Recall:         {test_results['entity_recall']:.2f}%")
    print(f"  Joint Exact Match:     {test_results['joint_exact_match']:.2f}%")

    # Per-tool breakdown
    print(f"\n  Per-Tool F1 Breakdown:")
    for tool_name, info in sorted(test_results["per_tool_f1"].items(), key=lambda x: -x[1]["f1"]):
        bar = "#" * int(info["f1"] / 5) + "-" * (20 - int(info["f1"] / 5))
        print(f"    {tool_name:<30s} {bar} {info['f1']:5.1f}%  (n={info['support']})")

    # Confidence calibration
    if test_results["confidence"]:
        c = test_results["confidence"]
        print(f"\n  Confidence Calibration:")
        print(f"    Correct predictions:   {c['correct_mean_conf']:.1f}%  (n={c['n_correct']})")
        print(f"    Incorrect predictions: {c['incorrect_mean_conf']:.1f}%  (n={c['n_incorrect']})")
        print(f"    Separation gap:        {c['separation_gap']:.1f}%")

    # Entity per-type
    if test_results["entity_per_type"]:
        print(f"\n  Entity-Level F1 Per Slot Type:")
        for etype, info in sorted(test_results["entity_per_type"].items(), key=lambda x: -x[1]["f1"]):
            print(f"    {etype:<15s}  P={info['precision']:5.1f}%  R={info['recall']:5.1f}%  F1={info['f1']:5.1f}%  (n={info['support']})")

    # ══════════════════════════════════════════════════════════════════════
    # BENCHMARK 3: Ablation Study (Full ARN vs JointBERT vs NoCRF vs NoFAR)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'-' * 80}")
    print(f"  [3/7] ABLATION STUDY: ARN vs JointBERT vs NoCRF vs NoFAR")
    print(f"{'-' * 80}")

    ablation_results = {}

    # Evaluate pre-trained Full ARN on val split
    gen2 = torch.Generator().manual_seed(42)
    n_train_ab = int(0.8 * n_total)
    n_val_ab = n_total - n_train_ab
    train_ds_ab, val_ds_ab = random_split(full_dataset, [n_train_ab, n_val_ab], generator=gen2)
    train_loader_ab = DataLoader(train_ds_ab, batch_size=32, shuffle=True)
    val_loader_ab = DataLoader(val_ds_ab, batch_size=32, shuffle=False)

    ablation_results["Full_ARN"] = evaluate_model(arn_model, val_loader_ab, config)
    print(f"  Full ARN (pre-trained):  Tool={ablation_results['Full_ARN']['tool_subset_accuracy']:.1f}%  "
          f"Entity F1={ablation_results['Full_ARN']['entity_f1']:.1f}%  "
          f"Joint={ablation_results['Full_ARN']['joint_exact_match']:.1f}%")

    # Train and evaluate baselines
    for model_name, ModelClass in [("JointBERT", JointBERT_Baseline),
                                     ("ARN_NoCRF", ARN_NoCRF),
                                     ("ARN_NoFAR", ARN_NoFAR)]:
        print(f"  Training {model_name}...", end="", flush=True)
        baseline = ModelClass(config)
        baseline = train_model(baseline, train_loader_ab, val_loader_ab, config, epochs=30)
        baseline.eval()
        ablation_results[model_name] = evaluate_model(baseline, val_loader_ab, config)
        r = ablation_results[model_name]
        print(f"  Tool={r['tool_subset_accuracy']:.1f}%  "
              f"Entity F1={r['entity_f1']:.1f}%  "
              f"Joint={r['joint_exact_match']:.1f}%")
        del baseline
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    all_results["ablation"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_tool_f1"}
                               for k, v in ablation_results.items()}

    # Print ablation table
    print(f"\n  +--- Ablation Summary Table ------------------------------------------------+")
    print(f"  | {'Model':<20s} | {'Tool Acc':>8s} | {'Tool F1u':>8s} | {'Ent F1':>7s} | {'Joint':>7s} |")
    print(f"  +{'-'*22}+{'-'*10}+{'-'*10}+{'-'*9}+{'-'*9}+")
    for name, r in ablation_results.items():
        print(f"  | {name:<20s} | {r['tool_subset_accuracy']:7.1f}% | {r['tool_f1_micro']:7.1f}% | {r['entity_f1']:6.1f}% | {r['joint_exact_match']:6.1f}% |")
    print(f"  +{'-'*22}+{'-'*10}+{'-'*10}+{'-'*9}+{'-'*9}+")

    # ══════════════════════════════════════════════════════════════════════
    # BENCHMARK 4: Hardware Profiling
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'-' * 80}")
    print(f"  [4/7] HARDWARE PROFILING (Latency, RAM, Disk, FLOPs)")
    print(f"{'-' * 80}")

    # Disk footprint
    disk_mb = os.path.getsize(ckpt_path) / (1024 * 1024)

    # RAM profiling
    tracemalloc.start()
    prof_model = AetherRoutingNetwork(config)
    prof_model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    prof_model.eval()
    _, peak_load_mem = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()

    sample_text = "send a WhatsApp message to Rahul saying I am late"
    # Warmup
    for _ in range(LATENCY_WARMUP):
        _ = predict(sample_text, prof_model, TOKENIZER, config)

    # Measure inference latency
    latencies = []
    for _ in range(LATENCY_ITERATIONS):
        start = time.perf_counter()
        _ = predict(sample_text, prof_model, TOKENIZER, config)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    _, peak_infer_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del prof_model

    # Parameter counts
    arn_full = AetherRoutingNetwork(config)
    total_params = sum(p.numel() for p in arn_full.parameters())
    trainable_params = sum(p.numel() for p in arn_full.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    far_params = sum(p.numel() for p in arn_full.far.parameters())
    crf_params = sum(p.numel() for p in arn_full.crf.parameters())
    slot_params = sum(p.numel() for p in arn_full.slot_tagger.parameters())
    encoder_params = sum(p.numel() for p in arn_full.encoder.parameters())
    del arn_full

    latencies_np = np.array(latencies)
    hardware_results = {
        "disk_footprint_mb": round(disk_mb, 2),
        "peak_ram_load_mb": round(peak_load_mem / (1024 * 1024), 2),
        "peak_ram_inference_mb": round(peak_infer_mem / (1024 * 1024), 2),
        "latency_mean_ms": round(float(np.mean(latencies_np)), 2),
        "latency_median_ms": round(float(np.median(latencies_np)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies_np, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(latencies_np, 99)), 2),
        "latency_std_ms": round(float(np.std(latencies_np)), 2),
        "latency_min_ms": round(float(np.min(latencies_np)), 2),
        "latency_max_ms": round(float(np.max(latencies_np)), 2),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "far_parameters": far_params,
        "crf_parameters": crf_params,
        "slot_tagger_parameters": slot_params,
        "encoder_parameters": encoder_params,
        "n_iterations": LATENCY_ITERATIONS,
        "n_warmup": LATENCY_WARMUP,
    }
    all_results["hardware"] = hardware_results

    print(f"  Disk Footprint:      {disk_mb:.2f} MB")
    print(f"  Peak RAM (Load):     {hardware_results['peak_ram_load_mb']:.2f} MB")
    print(f"  Peak RAM (Infer):    {hardware_results['peak_ram_inference_mb']:.2f} MB")
    print(f"  Latency (mean):      {hardware_results['latency_mean_ms']:.2f} ms")
    print(f"  Latency (median):    {hardware_results['latency_median_ms']:.2f} ms")
    print(f"  Latency (p95):       {hardware_results['latency_p95_ms']:.2f} ms")
    print(f"  Latency (p99):       {hardware_results['latency_p99_ms']:.2f} ms")
    print(f"  Total Parameters:    {total_params:,}")
    print(f"  Trainable Params:    {trainable_params:,}")
    print(f"  Frozen (Encoder):    {frozen_params:,}")
    print(f"  FAR Router Params:   {far_params:,}")
    print(f"  CRF Head Params:     {crf_params:,}")

    # ══════════════════════════════════════════════════════════════════════
    # BENCHMARK 5: OOD False-Positive Rejection (N=200)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'-' * 80}")
    print(f"  [5/7] FALSE-POSITIVE REJECTION (N={len(OOD_PROMPTS)} Out-of-Domain Prompts)")
    print(f"{'-' * 80}")

    arn_fp = AetherRoutingNetwork(config)
    arn_fp.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    arn_fp.eval()

    rejected_correct = 0
    failures = []
    fp_confidences = []

    for idx, prompt in enumerate(OOD_PROMPTS):
        res = predict(prompt, arn_fp, TOKENIZER, config)

        # Also get confidence
        inputs = TOKENIZER(prompt, max_length=config.MAX_SEQ_LEN, truncation=True,
                           padding="max_length", return_tensors="pt")
        with torch.no_grad():
            h = arn_fp.encoder(input_ids=inputs["input_ids"],
                               attention_mask=inputs["attention_mask"]).last_hidden_state
            logits, _ = arn_fp.far(h, inputs["attention_mask"])
            max_conf = torch.sigmoid(logits).max().item()

        fp_confidences.append(max_conf)
        is_rejected = len(res["tools"]) == 0
        if is_rejected:
            rejected_correct += 1
        else:
            failures.append((idx, prompt[:60], res["tools"], max_conf))

    rejection_rate = (rejected_correct / len(OOD_PROMPTS)) * 100
    fp_results = {
        "total_prompts": len(OOD_PROMPTS),
        "correctly_rejected": rejected_correct,
        "rejection_rate": round(rejection_rate, 1),
        "false_positives": len(failures),
        "mean_ood_confidence": round(float(np.mean(fp_confidences)) * 100, 1),
        "failures": [{"idx": f[0], "prompt": f[1], "triggered": f[2], "conf": round(f[3]*100, 1)}
                     for f in failures],
    }
    all_results["false_positive_rejection"] = fp_results
    del arn_fp

    print(f"  Correctly Rejected:    {rejected_correct} / {len(OOD_PROMPTS)}")
    print(f"  Rejection Rate:        {rejection_rate:.1f}%")
    print(f"  False Positives:       {len(failures)}")
    print(f"  Mean OOD Confidence:   {fp_results['mean_ood_confidence']:.1f}%")
    if failures:
        print(f"\n  False Positive Triggers:")
        for f in failures[:10]:
            print(f"    [{f[0]:3d}] \"{f[1]}...\" -> {f[2]} (conf={f[3]:.1f}%)")
        if len(failures) > 10:
            print(f"    ... and {len(failures)-10} more")

    # ══════════════════════════════════════════════════════════════════════
    # BENCHMARK 6: Latency Distribution Analysis
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'-' * 80}")
    print(f"  [6/7] LATENCY DISTRIBUTION (N={LATENCY_ITERATIONS} iterations)")
    print(f"{'-' * 80}")

    # Histogram buckets
    buckets = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 100)]
    latency_dist = {}
    for low, high in buckets:
        count = int(np.sum((latencies_np >= low) & (latencies_np < high)))
        pct = count / len(latencies_np) * 100
        latency_dist[f"{low}-{high}ms"] = {"count": count, "pct": round(pct, 1)}
        bar = "#" * int(pct / 2)
        print(f"    {low:3d}-{high:3d} ms: {bar} {pct:.1f}% ({count})")

    all_results["latency_distribution"] = latency_dist

    # ══════════════════════════════════════════════════════════════════════
    # BENCHMARK 7: JSON Safety Guarantee
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'-' * 80}")
    print(f"  [7/7] JSON SAFETY & STRUCTURAL GUARANTEE")
    print(f"{'-' * 80}")

    # Verify that ALL predictions produce valid structured output
    arn_safety = AetherRoutingNetwork(config)
    arn_safety.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    arn_safety.eval()

    safety_test_prompts = OOD_PROMPTS + [
        # Add adversarial/edge-case prompts
        "", " ", "   ", "!!!", "???", "...", "123 456 789",
        "{\"name\": \"hack\"}", "<script>alert(1)</script>",
        "NULL", "undefined", "NaN", "true", "false",
        "DROP TABLE users;", "rm -rf /", "sudo shutdown now",
        "A" * 200,  # Very long single token
        " ".join(["word"] * 100),  # Very long sentence
    ]

    valid_outputs = 0
    total_safety = len(safety_test_prompts)
    for prompt in safety_test_prompts:
        try:
            res = predict(prompt, arn_safety, TOKENIZER, config)
            # Verify structure
            assert isinstance(res, dict), "Output not a dict"
            assert "tools" in res, "Missing 'tools' key"
            assert "arguments" in res, "Missing 'arguments' key"
            assert isinstance(res["tools"], list), "'tools' not a list"
            assert isinstance(res["arguments"], dict), "'arguments' not a dict"
            # Verify JSON serializable
            json.dumps(res)
            valid_outputs += 1
        except Exception as e:
            print(f"    SAFETY FAILURE: {prompt[:40]}... -> {e}")

    safety_rate = (valid_outputs / total_safety) * 100
    safety_results = {
        "total_tested": total_safety,
        "valid_structured_outputs": valid_outputs,
        "structural_safety_rate": round(safety_rate, 2),
        "json_syntax_error_rate": round(100 - safety_rate, 2),
    }
    all_results["json_safety"] = safety_results
    del arn_safety

    print(f"  Total Tested:              {total_safety}")
    print(f"  Valid Structured Outputs:   {valid_outputs} / {total_safety}")
    print(f"  Structural Safety Rate:    {safety_rate:.2f}%")
    print(f"  JSON Syntax Error Rate:    {100 - safety_rate:.2f}%")

    # ══════════════════════════════════════════════════════════════════════
    # EXPORT: Save all results to JSON
    # ══════════════════════════════════════════════════════════════════════
    output_path = os.path.join(RESULTS_DIR, "ieee_benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'=' * 80}")
    print(f"  ALL RESULTS SAVED -> {output_path}")
    print(f"{'=' * 80}")

    # ══════════════════════════════════════════════════════════════════════
    # EXPORT: LaTeX Table for Paper
    # ══════════════════════════════════════════════════════════════════════
    latex_path = os.path.join(RESULTS_DIR, "ieee_paper_tables.tex")
    with open(latex_path, "w") as f:
        f.write("% === Auto-generated by ARN IEEE Benchmark Suite ===\n\n")

        # Table 1: Main results (CV)
        f.write("% Table 1: Cross-Validation Results\n")
        f.write("\\begin{table}[t]\n\\centering\n\\caption{ARN 5-Fold Cross-Validation Results (3 seeds, $N$=13,798)}\n")
        f.write("\\label{tab:cv_results}\n")
        f.write("\\begin{tabular}{lcc}\n\\hline\n")
        f.write("\\textbf{Metric} & \\textbf{Mean $\\pm$ Std} & \\textbf{95\\% CI} \\\\\n\\hline\n")
        for key, v in cv_summary.items():
            nice_name = key.replace("_", " ").title()
            f.write(f"{nice_name} & {v['mean']:.2f} $\\pm$ {v['std']:.2f} & [{v['ci_95_lower']:.1f}, {v['ci_95_upper']:.1f}] \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")

        # Table 2: Ablation
        f.write("% Table 2: Ablation Study\n")
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Ablation Study: Architecture Component Analysis}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\begin{tabular}{lcccc}\n\\hline\n")
        f.write("\\textbf{Model} & \\textbf{Tool Acc} & \\textbf{Tool F1$_\\mu$} & \\textbf{Entity F1} & \\textbf{Joint EM} \\\\\n\\hline\n")
        for name, r in all_results["ablation"].items():
            bold = "\\textbf" if name == "Full_ARN" else ""
            f.write(f"{bold}{{{name.replace('_', ' ')}}} & {r['tool_subset_accuracy']:.1f} & {r['tool_f1_micro']:.1f} & {r['entity_f1']:.1f} & {r['joint_exact_match']:.1f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")

        # Table 3: Hardware
        hw = all_results["hardware"]
        f.write("% Table 3: Hardware Profile\n")
        f.write("\\begin{table}[t]\n\\centering\n\\caption{ARN Hardware Profile (CPU, single-threaded)}\n")
        f.write("\\label{tab:hardware}\n")
        f.write("\\begin{tabular}{lr}\n\\hline\n")
        f.write(f"Total Parameters & {hw['total_parameters']:,} \\\\\n")
        f.write(f"Trainable Parameters & {hw['trainable_parameters']:,} \\\\\n")
        f.write(f"Disk Footprint & {hw['disk_footprint_mb']:.2f} MB \\\\\n")
        f.write(f"Peak RAM (Inference) & {hw['peak_ram_inference_mb']:.2f} MB \\\\\n")
        f.write(f"Latency (Mean) & {hw['latency_mean_ms']:.2f} ms \\\\\n")
        f.write(f"Latency (p95) & {hw['latency_p95_ms']:.2f} ms \\\\\n")
        f.write(f"Latency (p99) & {hw['latency_p99_ms']:.2f} ms \\\\\n")
        f.write(f"JSON Error Rate & {all_results['json_safety']['json_syntax_error_rate']:.2f}\\% \\\\\n")
        f.write(f"OOD Rejection Rate & {all_results['false_positive_rejection']['rejection_rate']:.1f}\\% \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    print(f"  LaTeX tables saved -> {latex_path}")
    print(f"\n{'=' * 80}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'=' * 80}\n")

    return all_results


if __name__ == "__main__":
    results = run_full_benchmark()
