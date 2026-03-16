#!/usr/bin/env python3
"""
Formula Detector for Hinge Openers
====================================
Builds a mathematical detection system using sentence embeddings to identify
and penalize formulaic openers. Uses LDA-style direction vectors, template
similarity, and word-level features to produce a composite "formula score".

Usage:
    python3 -u formula_detector.py          # Train + validate + export model
    python3 -u formula_detector.py --score "your opener here"  # Score a single opener

Model is saved to formula_detector_model.json for reuse.
"""

import json
import re
import sys
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent
EVAL_GLOB = str(SRC_DIR / "eval_results_gen_*.json")
MODEL_FILE = SRC_DIR / "formula_detector_model.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SCORE_DIMS = [
    "reply_likelihood", "humor_quality", "creativity", "specificity",
    "human_likeness", "closer_strength", "warmth_signal", "formula_freshness"
]

# Thresholds for ground truth labels
FORMULAIC_THRESHOLD = 4   # freshness <= this => formulaic
FRESH_THRESHOLD = 8       # freshness >= this => fresh

# Template phrases known to indicate formulaic patterns
TEMPLATE_PHRASES = [
    "is just",
    "who hurt you",
    "what did you do",
    "that's not X that's Y",
    "coffee or boba",
    "who's your agent",
    "who's their agent",
    "who's your publicist",
    "who's their publicist",
    "what's the conversion rate",
    "bribery with extra steps",
    "we should test it over coffee",
    "how many reps is that",
    "threat level",
]

# ─── HELPERS ──────────────────────────────────────────────────────────
def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    return 1.0 - cosine_dist(a, b)

def normalize(v):
    """L2-normalize a vector."""
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

SELF_INSERT_RE = re.compile(
    r"\bi[''`]?m\b|\bmy\b|\bi have\b|\bi was\b|\bi used to\b|\bi can\b|\bi['`]?ll\b|\bi\b",
    re.IGNORECASE
)

def has_word_just(text):
    return bool(re.search(r'\bjust\b', text, re.IGNORECASE))

def has_is_just(text):
    return bool(re.search(r'\bis just\b|\bare just\b', text, re.IGNORECASE))

def has_self_insert(text):
    return bool(SELF_INSERT_RE.search(text))

def word_count(text):
    return len(text.split())

# ─── DATA LOADING ────────────────────────────────────────────────────
def load_all_eval_data():
    """Load all openers with formula_freshness scores from all eval files."""
    files = sorted(glob.glob(EVAL_GLOB))
    seen = set()
    openers = []

    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        for r in data["results"]:
            if "formula_freshness" not in r["scores"]:
                continue
            text = r["comment"]
            if text in seen:
                continue
            seen.add(text)

            scores = {}
            for dim in SCORE_DIMS:
                if dim in r["scores"]:
                    scores[dim] = r["scores"][dim]["score"]
            scores["avg"] = float(np.mean(list(scores.values())))

            openers.append({
                "text": text,
                "model": r.get("model", ""),
                "scores": scores,
            })

    return openers

# ─── MAIN ANALYSIS ───────────────────────────────────────────────────
def train_detector():
    """Train the formula detector and return the model parameters."""

    print("=" * 80)
    print("FORMULA DETECTOR: TRAINING PIPELINE")
    print("=" * 80)

    # ── Step 0: Load data ──
    openers = load_all_eval_data()
    print(f"\nLoaded {len(openers)} unique openers with formula_freshness scores")

    freshness = np.array([o["scores"]["formula_freshness"] for o in openers])
    formulaic_idx = [i for i, f in enumerate(freshness) if f <= FORMULAIC_THRESHOLD]
    fresh_idx = [i for i, f in enumerate(freshness) if f >= FRESH_THRESHOLD]
    mid_idx = [i for i, f in enumerate(freshness)
               if FORMULAIC_THRESHOLD < f < FRESH_THRESHOLD]

    print(f"  Formulaic (freshness <= {FORMULAIC_THRESHOLD}): {len(formulaic_idx)}")
    print(f"  Middle    ({FORMULAIC_THRESHOLD} < freshness < {FRESH_THRESHOLD}): {len(mid_idx)}")
    print(f"  Fresh     (freshness >= {FRESH_THRESHOLD}): {len(fresh_idx)}")

    # ── Step 1: Embed everything ──
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embed = lambda texts: model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    texts = [o["text"] for o in openers]
    embeddings = embed(texts)
    print(f"Embedded {len(texts)} openers -> shape {embeddings.shape}")

    # ══════════════════════════════════════════════════════════════
    # SECTION 1: FORMULA DIRECTION VECTOR (LDA-style)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SECTION 1: FORMULA DIRECTION VECTOR")
    print("=" * 80)

    formulaic_embs = embeddings[formulaic_idx]
    fresh_embs = embeddings[fresh_idx]

    centroid_formulaic = formulaic_embs.mean(axis=0)
    centroid_fresh = fresh_embs.mean(axis=0)

    # Direction: formulaic - fresh (positive = more formulaic)
    formula_direction = centroid_formulaic - centroid_fresh
    formula_direction_norm = normalize(formula_direction)

    centroid_sim = cosine_sim(
        normalize(centroid_formulaic), normalize(centroid_fresh)
    )
    print(f"\n  Centroid cosine similarity (formulaic vs fresh): {centroid_sim:.4f}")
    print(f"  Centroid separation (L2): {np.linalg.norm(formula_direction):.4f}")

    # Project all openers onto formula direction
    projections = embeddings @ formula_direction_norm
    proj_r, proj_p = spearmanr(projections, freshness)
    print(f"\n  Projection onto formula direction vs freshness:")
    print(f"    Spearman r = {proj_r:+.4f}, p = {proj_p:.6f}")
    print(f"    (Negative r expected: high projection = formulaic = low freshness)")

    # Find optimal threshold on projection
    print(f"\n  Threshold search on formula direction projection:")
    print(f"    {'Threshold':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Acc':>10s}")

    best_f1 = 0
    best_proj_threshold = 0
    for pct in range(5, 96, 5):
        thresh = np.percentile(projections, pct)
        # Predict "formulaic" if projection >= threshold
        pred_formulaic = projections >= thresh
        true_formulaic = freshness <= FORMULAIC_THRESHOLD

        tp = np.sum(pred_formulaic & true_formulaic)
        fp = np.sum(pred_formulaic & ~true_formulaic)
        fn = np.sum(~pred_formulaic & true_formulaic)
        tn = np.sum(~pred_formulaic & ~true_formulaic)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        acc = (tp + tn) / len(openers)

        if f1 > best_f1:
            best_f1 = f1
            best_proj_threshold = thresh

        print(f"    {thresh:10.4f} {precision:10.3f} {recall:10.3f} {f1:10.3f} {acc:10.3f}")

    print(f"\n  Best projection threshold: {best_proj_threshold:.4f} (F1={best_f1:.3f})")

    # ══════════════════════════════════════════════════════════════
    # SECTION 2: COSINE SIMILARITY TO FORMULAIC CENTROID
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SECTION 2: COSINE SIMILARITY TO FORMULAIC CENTROID")
    print("=" * 80)

    centroid_formulaic_normed = normalize(centroid_formulaic)
    cosine_to_formulaic = embeddings @ centroid_formulaic_normed

    cos_r, cos_p = spearmanr(cosine_to_formulaic, freshness)
    print(f"\n  Cosine sim to formulaic centroid vs freshness:")
    print(f"    Spearman r = {cos_r:+.4f}, p = {cos_p:.6f}")

    # Threshold search
    print(f"\n  Threshold search on cosine to formulaic centroid:")
    print(f"    {'Threshold':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")

    best_cos_f1 = 0
    best_cos_threshold = 0
    for pct in range(50, 96, 5):
        thresh = np.percentile(cosine_to_formulaic, pct)
        pred_formulaic = cosine_to_formulaic >= thresh
        true_formulaic = freshness <= FORMULAIC_THRESHOLD

        tp = np.sum(pred_formulaic & true_formulaic)
        fp = np.sum(pred_formulaic & ~true_formulaic)
        fn = np.sum(~pred_formulaic & true_formulaic)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_cos_f1:
            best_cos_f1 = f1
            best_cos_threshold = thresh

        print(f"    {thresh:10.4f} {precision:10.3f} {recall:10.3f} {f1:10.3f}")

    print(f"\n  Best cosine threshold: {best_cos_threshold:.4f} (F1={best_cos_f1:.3f})")

    # ══════════════════════════════════════════════════════════════
    # SECTION 3: TEMPLATE PHRASE SIMILARITY
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SECTION 3: TEMPLATE PHRASE SIMILARITY")
    print("=" * 80)

    template_embs = embed(TEMPLATE_PHRASES)
    print(f"\n  Embedded {len(TEMPLATE_PHRASES)} template phrases")

    # For each opener, compute similarity to each template
    # Shape: (n_openers, n_templates)
    template_sims = embeddings @ template_embs.T

    # Max similarity across all templates
    max_template_sim = template_sims.max(axis=1)

    max_sim_r, max_sim_p = spearmanr(max_template_sim, freshness)
    print(f"\n  Max template similarity vs freshness:")
    print(f"    Spearman r = {max_sim_r:+.4f}, p = {max_sim_p:.6f}")

    # Per-template analysis
    print(f"\n  Per-template correlation with freshness:")
    template_correlations = []
    for t, phrase in enumerate(TEMPLATE_PHRASES):
        r, p = spearmanr(template_sims[:, t], freshness)
        template_correlations.append((phrase, r, p, t))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {phrase:40s} r={r:+.4f} p={p:.6f} {sig}")

    # Sort by predictive power
    template_correlations.sort(key=lambda x: x[1])
    print(f"\n  Most predictive templates (negative r = predicts formulaic):")
    for phrase, r, p, t in template_correlations[:5]:
        print(f"    r={r:+.4f} \"{phrase}\"")

    # Best template threshold search
    print(f"\n  Threshold search on max template similarity:")
    print(f"    {'Threshold':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")

    best_tmpl_f1 = 0
    best_tmpl_threshold = 0
    for pct in range(50, 96, 5):
        thresh = np.percentile(max_template_sim, pct)
        pred_formulaic = max_template_sim >= thresh
        true_formulaic = freshness <= FORMULAIC_THRESHOLD

        tp = np.sum(pred_formulaic & true_formulaic)
        fp = np.sum(pred_formulaic & ~true_formulaic)
        fn = np.sum(~pred_formulaic & true_formulaic)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_tmpl_f1:
            best_tmpl_f1 = f1
            best_tmpl_threshold = thresh

        print(f"    {thresh:10.4f} {precision:10.3f} {recall:10.3f} {f1:10.3f}")

    print(f"\n  Best template threshold: {best_tmpl_threshold:.4f} (F1={best_tmpl_f1:.3f})")

    # ══════════════════════════════════════════════════════════════
    # SECTION 4: WORD-LEVEL FEATURES
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SECTION 4: WORD-LEVEL FEATURES")
    print("=" * 80)

    feat_just = np.array([float(has_word_just(o["text"])) for o in openers])
    feat_is_just = np.array([float(has_is_just(o["text"])) for o in openers])
    feat_self = np.array([float(has_self_insert(o["text"])) for o in openers])
    feat_wc = np.array([word_count(o["text"]) for o in openers])

    features = {
        "has_just": feat_just,
        "has_is_just": feat_is_just,
        "has_self_insert": feat_self,
        "word_count": feat_wc,
    }

    print(f"\n  Word-level feature correlations with freshness:")
    for name, feat in features.items():
        r, p = spearmanr(feat, freshness)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {name:20s}: r={r:+.4f} p={p:.6f} {sig}")

    # Feature distribution by class
    print(f"\n  Feature prevalence by class:")
    for name, feat in features.items():
        form_rate = feat[formulaic_idx].mean()
        fresh_rate = feat[fresh_idx].mean()
        print(f"    {name:20s}: formulaic={form_rate:.3f}  fresh={fresh_rate:.3f}  "
              f"ratio={form_rate/fresh_rate:.2f}x" if fresh_rate > 0 else
              f"    {name:20s}: formulaic={form_rate:.3f}  fresh={fresh_rate:.3f}")

    # ══════════════════════════════════════════════════════════════
    # SECTION 5: COMPOSITE SCORE OPTIMIZATION
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SECTION 5: COMPOSITE SCORE OPTIMIZATION")
    print("=" * 80)

    # Features for the composite model:
    # x1 = projection onto formula direction (higher = more formulaic)
    # x2 = max template similarity (higher = more formulaic)
    # x3 = has_is_just (1 if contains "is just" / "are just" -- 61x formulaic ratio)
    # x4 = 1 - has_self_insert (lack of self = more formulaic)
    # x5 = cosine_to_formulaic centroid
    # x6 = has_just (1 if contains "just" anywhere -- 3.9x ratio)

    x1 = projections
    x2 = max_template_sim
    x3 = feat_is_just
    x4 = 1.0 - feat_self
    x5 = cosine_to_formulaic
    x6 = feat_just

    # Normalize each feature to [0, 1] for interpretable weights
    def minmax_normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    x1_norm = minmax_normalize(x1)
    x2_norm = minmax_normalize(x2)
    x3_norm = x3  # Already 0/1
    x4_norm = x4  # Already 0/1
    x5_norm = minmax_normalize(x5)
    x6_norm = x6  # Already 0/1

    feature_matrix = np.column_stack([x1_norm, x2_norm, x3_norm, x4_norm, x5_norm, x6_norm])
    feature_names = [
        "formula_direction_proj",
        "max_template_sim",
        "has_is_just",
        "lacks_self_insert",
        "cosine_to_formulaic",
        "has_just",
    ]

    # ── Approach 1: Logistic Regression (principled baseline) ──
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, f1_score

    true_formulaic = freshness <= FORMULAIC_THRESHOLD
    true_labels = true_formulaic.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)

    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_scaled, true_labels)
    lr_probs = lr.predict_proba(X_scaled)[:, 1]
    lr_pred = lr.predict(X_scaled)

    print(f"\n  Logistic Regression (class_weight='balanced'):")
    print(f"    Coefficients:")
    for i, name in enumerate(feature_names):
        print(f"      {name:30s}: {lr.coef_[0][i]:+.4f}")
    print(f"    Intercept: {lr.intercept_[0]:+.4f}")
    lr_f1 = f1_score(true_labels, lr_pred)
    print(f"    F1 (on training data): {lr_f1:.3f}")

    # ── Approach 2: Grid search for interpretable linear model ──
    print(f"\n  Grid search over weight combinations...")
    print(f"  Features: {feature_names}")

    best_weights = None
    best_composite_f1 = 0
    best_composite_threshold = 0
    best_results = None

    # Individual feature performance
    individual_f1s = []
    for fi in range(len(feature_names)):
        feat_vals = feature_matrix[:, fi]
        best_fi_f1 = 0
        for pct in range(10, 96, 3):
            thresh = np.percentile(feat_vals, pct)
            pred = feat_vals >= thresh
            tp = np.sum(pred & true_formulaic)
            fp = np.sum(pred & ~true_formulaic)
            fn = np.sum(~pred & true_formulaic)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best_fi_f1:
                best_fi_f1 = f1
        individual_f1s.append(best_fi_f1)
        print(f"    Feature '{feature_names[fi]}' alone: best F1 = {best_fi_f1:.3f}")

    # Coarse grid: 6 features, use steps of 0.1 but limit to 4 values each
    # to keep search tractable (4^6 = 4096 combos)
    weight_candidates = [0.0, 0.15, 0.3, 0.5]
    n_feats = len(feature_names)
    print(f"\n  Running coarse grid search ({len(weight_candidates)}^{n_feats} = "
          f"{len(weight_candidates)**n_feats} weight combos)...")

    def evaluate_weights(ws, feature_mat, true_form):
        composite = feature_mat @ ws
        best_f1 = 0
        best_thresh = 0
        best_res = None
        for pct in range(40, 96, 3):
            thresh = np.percentile(composite, pct)
            pred = composite >= thresh
            tp = np.sum(pred & true_form)
            fp = np.sum(pred & ~true_form)
            fn = np.sum(~pred & true_form)
            p_val = tp / (tp + fp) if (tp + fp) > 0 else 0
            r_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_res = {"precision": p_val, "recall": r_val, "f1": f1,
                            "tp": int(tp), "fp": int(fp), "fn": int(fn)}
        return best_f1, best_thresh, best_res

    import itertools
    search_count = 0
    for combo in itertools.product(weight_candidates, repeat=n_feats):
        ws = np.array(combo)
        if ws.sum() < 0.1:
            continue
        ws = ws / ws.sum()
        f1, thresh, res = evaluate_weights(ws, feature_matrix, true_formulaic)
        if f1 > best_composite_f1:
            best_composite_f1 = f1
            best_weights = ws.copy()
            best_composite_threshold = thresh
            best_results = res
        search_count += 1

    print(f"  Coarse search: {search_count} combos, best F1 = {best_composite_f1:.3f}")

    # Fine grid around best weights: perturb each weight individually and in pairs
    print(f"  Running fine-tuning around best weights...")
    fine_count = 0
    improved = True
    while improved:
        improved = False
        # Perturb each weight individually
        for fi in range(n_feats):
            for delta in np.arange(-0.15, 0.16, 0.01):
                ws = best_weights.copy()
                ws[fi] += delta
                ws = np.clip(ws, 0, None)
                if ws.sum() < 0.1:
                    continue
                ws = ws / ws.sum()
                f1, thresh, res = evaluate_weights(ws, feature_matrix, true_formulaic)
                if f1 > best_composite_f1:
                    best_composite_f1 = f1
                    best_weights = ws.copy()
                    best_composite_threshold = thresh
                    best_results = res
                    improved = True
                fine_count += 1
        # Perturb pairs
        for fi in range(n_feats):
            for fj in range(fi + 1, n_feats):
                for di in np.arange(-0.1, 0.11, 0.025):
                    for dj in np.arange(-0.1, 0.11, 0.025):
                        ws = best_weights.copy()
                        ws[fi] += di
                        ws[fj] += dj
                        ws = np.clip(ws, 0, None)
                        if ws.sum() < 0.1:
                            continue
                        ws = ws / ws.sum()
                        f1, thresh, res = evaluate_weights(ws, feature_matrix, true_formulaic)
                        if f1 > best_composite_f1:
                            best_composite_f1 = f1
                            best_weights = ws.copy()
                            best_composite_threshold = thresh
                            best_results = res
                            improved = True
                        fine_count += 1

    print(f"  Fine search: {fine_count} evaluations, best F1 = {best_composite_f1:.3f}")

    # Also try the LR-derived weights (use absolute coefficient magnitudes as weights)
    lr_abs = np.abs(lr.coef_[0])
    lr_weights = lr_abs / lr_abs.sum()
    lr_f1_grid, lr_thresh, lr_res = evaluate_weights(lr_weights, feature_matrix, true_formulaic)
    print(f"  LR-derived weights F1: {lr_f1_grid:.3f}")
    if lr_f1_grid > best_composite_f1:
        best_composite_f1 = lr_f1_grid
        best_weights = lr_weights.copy()
        best_composite_threshold = lr_thresh
        best_results = lr_res
        print(f"  --> LR weights are better, adopting them")

    print(f"  Searched {search_count + fine_count} total weight combinations")
    print(f"\n  *** OPTIMAL WEIGHTS ***")
    for i, name in enumerate(feature_names):
        print(f"    w_{name} = {best_weights[i]:.4f}")
    print(f"\n  Optimal threshold: {best_composite_threshold:.4f}")
    print(f"  F1 = {best_composite_f1:.3f}")
    print(f"  Precision = {best_results['precision']:.3f}")
    print(f"  Recall = {best_results['recall']:.3f}")
    print(f"  TP={best_results['tp']}, FP={best_results['fp']}, FN={best_results['fn']}")

    # Compute final composite scores with optimal weights
    composite_scores = feature_matrix @ best_weights

    # ── Fine-tune threshold for multiple operating points ──
    print(f"\n  Operating points at optimal weights:")
    print(f"    {'Threshold':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'FPR':>10s}")

    operating_points = []
    for pct in range(30, 98, 2):
        thresh = np.percentile(composite_scores, pct)
        pred = composite_scores >= thresh
        tp = np.sum(pred & true_formulaic)
        fp = np.sum(pred & ~true_formulaic)
        fn = np.sum(~pred & true_formulaic)
        tn = np.sum(~pred & ~true_formulaic)

        p_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        r_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        operating_points.append((thresh, p_val, r_val, f1, fpr))
        if pct % 10 == 0 or f1 > best_composite_f1 - 0.02:
            print(f"    {thresh:10.4f} {p_val:10.3f} {r_val:10.3f} {f1:10.3f} {fpr:10.3f}")

    # Find high-precision operating point (for production use: reject with high confidence)
    high_prec_points = [(t, p, r, f, fpr) for t, p, r, f, fpr in operating_points if p >= 0.5]
    if high_prec_points:
        hp_point = max(high_prec_points, key=lambda x: x[3])
        print(f"\n  Recommended production threshold (precision >= 0.5):")
        print(f"    threshold={hp_point[0]:.4f}, P={hp_point[1]:.3f}, R={hp_point[2]:.3f}, "
              f"F1={hp_point[3]:.3f}, FPR={hp_point[4]:.3f}")

    # ══════════════════════════════════════════════════════════════
    # SECTION 6: VALIDATION
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SECTION 6: VALIDATION")
    print("=" * 80)

    # Correlation of composite score with actual freshness
    comp_r, comp_p = spearmanr(composite_scores, freshness)
    comp_pr, comp_pp = pearsonr(composite_scores, freshness)
    print(f"\n  Composite score vs freshness:")
    print(f"    Spearman r = {comp_r:+.4f}, p = {comp_p:.6f}")
    print(f"    Pearson  r = {comp_pr:+.4f}, p = {comp_pp:.6f}")

    # Compare with individual features
    print(f"\n  Feature comparison (Spearman r with freshness):")
    all_features = {
        "composite": composite_scores,
        "projection": projections,
        "cosine_to_formulaic": cosine_to_formulaic,
        "max_template_sim": max_template_sim,
        "has_just": feat_just,
        "lacks_self_insert": 1.0 - feat_self,
    }
    for name, vals in all_features.items():
        r, p = spearmanr(vals, freshness)
        print(f"    {name:25s}: r={r:+.4f} p={p:.6f}")

    # Show correctly identified formulaic openers
    pred_at_threshold = composite_scores >= best_composite_threshold

    true_positives = [i for i in range(len(openers))
                      if pred_at_threshold[i] and true_formulaic[i]]
    false_positives = [i for i in range(len(openers))
                       if pred_at_threshold[i] and not true_formulaic[i]]
    false_negatives = [i for i in range(len(openers))
                       if not pred_at_threshold[i] and true_formulaic[i]]

    print(f"\n  Correctly caught formulaic openers (True Positives, showing up to 15):")
    for idx in sorted(true_positives, key=lambda i: composite_scores[i], reverse=True)[:15]:
        o = openers[idx]
        print(f"    score={composite_scores[idx]:.3f} fresh={o['scores']['formula_freshness']} "
              f"\"{o['text'][:85]}\"")

    print(f"\n  False Positives (flagged but actually fresh, showing up to 10):")
    for idx in sorted(false_positives, key=lambda i: composite_scores[i], reverse=True)[:10]:
        o = openers[idx]
        print(f"    score={composite_scores[idx]:.3f} fresh={o['scores']['formula_freshness']} "
              f"\"{o['text'][:85]}\"")

    print(f"\n  Missed formulaic openers (False Negatives, showing up to 10):")
    for idx in sorted(false_negatives, key=lambda i: composite_scores[i])[:10]:
        o = openers[idx]
        print(f"    score={composite_scores[idx]:.3f} fresh={o['scores']['formula_freshness']} "
              f"\"{o['text'][:85]}\"")

    # Score distribution by freshness bucket
    print(f"\n  Composite score distribution by freshness bucket:")
    for bucket_lo in range(1, 11):
        bucket_idx = [i for i, f in enumerate(freshness) if f == bucket_lo]
        if not bucket_idx:
            continue
        scores_in_bucket = composite_scores[bucket_idx]
        print(f"    freshness={bucket_lo:2d}: n={len(bucket_idx):4d}  "
              f"mean={scores_in_bucket.mean():.3f}  "
              f"std={scores_in_bucket.std():.3f}  "
              f"[{scores_in_bucket.min():.3f}, {scores_in_bucket.max():.3f}]")

    # ══════════════════════════════════════════════════════════════
    # SECTION 7: EXPORT MODEL
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SECTION 7: EXPORT MODEL")
    print("=" * 80)

    # Store normalization parameters for each continuous feature
    norm_params = {
        "x1_proj_min": float(x1.min()),
        "x1_proj_max": float(x1.max()),
        "x2_tmpl_min": float(x2.min()),
        "x2_tmpl_max": float(x2.max()),
        "x5_cos_min": float(x5.min()),
        "x5_cos_max": float(x5.max()),
    }
    # x3 (has_is_just), x4 (lacks_self_insert), x6 (has_just) are binary -- no normalization needed

    model_data = {
        "description": "Formula detector model for Hinge openers",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": int(embeddings.shape[1]),

        # Formula direction vector (384-dim)
        "formula_direction": formula_direction_norm.tolist(),

        # Formulaic centroid (normalized, 384-dim)
        "formulaic_centroid": centroid_formulaic_normed.tolist(),

        # Template phrases and their embeddings
        "template_phrases": TEMPLATE_PHRASES,
        "template_embeddings": template_embs.tolist(),

        # Composite model weights
        "feature_names": feature_names,
        "weights": best_weights.tolist(),

        # Normalization parameters (min/max for each continuous feature)
        "normalization": norm_params,

        # Threshold
        "threshold": float(best_composite_threshold),
        "threshold_metrics": {
            "precision": float(best_results["precision"]),
            "recall": float(best_results["recall"]),
            "f1": float(best_results["f1"]),
        },

        # Training stats
        "training_stats": {
            "n_openers": len(openers),
            "n_formulaic": len(formulaic_idx),
            "n_fresh": len(fresh_idx),
            "formulaic_threshold": FORMULAIC_THRESHOLD,
            "fresh_threshold": FRESH_THRESHOLD,
        },
    }

    with open(MODEL_FILE, "w") as f:
        json.dump(model_data, f, indent=2)

    file_size_kb = MODEL_FILE.stat().st_size / 1024
    print(f"\n  Model saved to: {MODEL_FILE}")
    print(f"  File size: {file_size_kb:.1f} KB")
    print(f"  Contents: formula direction vector ({embeddings.shape[1]}d), "
          f"formulaic centroid ({embeddings.shape[1]}d), "
          f"{len(TEMPLATE_PHRASES)} template embeddings, "
          f"{len(feature_names)} feature weights")

    # ── Logistic regression details (for reference) ──
    print(f"\n  Logistic Regression model (for comparison):")
    print(f"    LR coefficients: {lr.coef_[0].tolist()}")
    print(f"    LR intercept: {lr.intercept_[0]:.4f}")
    print(f"    LR scaler means: {scaler.mean_.tolist()}")
    print(f"    LR scaler stds: {scaler.scale_.tolist()}")

    # ══════════════════════════════════════════════════════════════
    # SECTION 8: FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("FINAL SUMMARY: MATHEMATICAL FORMULA")
    print("=" * 80)

    print(f"""
  THE FORMULA DETECTION MODEL
  ===========================

  For a candidate opener text T:

  1. Embed T using {EMBEDDING_MODEL} -> e (384-dim, L2-normalized)

  2. Compute 6 features:
     x1 = (e . formula_direction - {norm_params['x1_proj_min']:.4f}) / {norm_params['x1_proj_max'] - norm_params['x1_proj_min']:.4f}
          (projection onto formulaic-vs-fresh direction, min-max normalized)
     x2 = (max_t(e . template_t) - {norm_params['x2_tmpl_min']:.4f}) / {norm_params['x2_tmpl_max'] - norm_params['x2_tmpl_min']:.4f}
          (max cosine sim to {len(TEMPLATE_PHRASES)} template phrases, normalized)
     x3 = 1 if "is just" or "are just" in T else 0
          (the strongest single signal -- 61x formulaic ratio)
     x4 = 0 if T contains self-insertion ("I", "my", "I'm") else 1
          (lack of personal warmth signal)
     x5 = (e . formulaic_centroid - {norm_params['x5_cos_min']:.4f}) / {norm_params['x5_cos_max'] - norm_params['x5_cos_min']:.4f}
          (cosine similarity to formulaic cluster center, normalized)
     x6 = 1 if "just" in T else 0
          (toxic word flag, weaker but broader than x3)

  3. Compute composite formula score:
     formula_score = {best_weights[0]:.4f} * x1
                   + {best_weights[1]:.4f} * x2
                   + {best_weights[2]:.4f} * x3
                   + {best_weights[3]:.4f} * x4
                   + {best_weights[4]:.4f} * x5
                   + {best_weights[5]:.4f} * x6

  4. Decision:
     if formula_score >= {best_composite_threshold:.4f}:
         REJECT (formulaic opener detected)
     else:
         ACCEPT

  Performance at this threshold:
     Precision = {best_results['precision']:.3f}
     Recall    = {best_results['recall']:.3f}
     F1        = {best_results['f1']:.3f}

  Trained on {len(openers)} openers ({len(formulaic_idx)} formulaic, {len(fresh_idx)} fresh)
  Composite score correlation with freshness: Spearman r = {comp_r:+.4f}
""")

    return model_data


# ─── SCORER FUNCTION ──────────────────────────────────────────────────
def score_opener(text: str, model_path: str = None) -> dict:
    """
    Score a single opener for formulaic-ness.
    Loads the pre-trained model and returns a breakdown.

    Returns:
        {
            "formula_score": float,  # 0-1, higher = more formulaic
            "is_formulaic": bool,    # True if above threshold
            "features": {
                "formula_direction_proj": float,
                "max_template_sim": float,
                "has_just": float,
                "lacks_self_insert": float,
                "cosine_to_formulaic": float,
            },
            "closest_template": str,
            "closest_template_sim": float,
        }
    """
    if model_path is None:
        model_path = str(MODEL_FILE)

    with open(model_path) as f:
        md = json.load(f)

    # Load embedding model
    emb_model = SentenceTransformer(md["embedding_model"])
    emb = emb_model.encode([text], normalize_embeddings=True)[0]

    # Feature 1: projection onto formula direction
    formula_dir = np.array(md["formula_direction"])
    raw_proj = float(np.dot(emb, formula_dir))
    norm = md["normalization"]
    x1 = (raw_proj - norm["x1_proj_min"]) / (norm["x1_proj_max"] - norm["x1_proj_min"])
    x1 = np.clip(x1, 0, 1)

    # Feature 2: max template similarity
    template_embs = np.array(md["template_embeddings"])
    template_sims = emb @ template_embs.T
    raw_max_sim = float(template_sims.max())
    best_template_idx = int(template_sims.argmax())
    x2 = (raw_max_sim - norm["x2_tmpl_min"]) / (norm["x2_tmpl_max"] - norm["x2_tmpl_min"])
    x2 = np.clip(x2, 0, 1)

    # Feature 3: has_is_just (strongest single signal)
    x3 = 1.0 if has_is_just(text) else 0.0

    # Feature 4: lacks self-insertion
    x4 = 0.0 if has_self_insert(text) else 1.0

    # Feature 5: cosine to formulaic centroid
    formulaic_centroid = np.array(md["formulaic_centroid"])
    raw_cos = float(np.dot(emb, formulaic_centroid))
    x5 = (raw_cos - norm["x5_cos_min"]) / (norm["x5_cos_max"] - norm["x5_cos_min"])
    x5 = np.clip(x5, 0, 1)

    # Feature 6: has_just (broader signal)
    x6 = 1.0 if has_word_just(text) else 0.0

    # Composite score
    weights = np.array(md["weights"])
    features_vec = np.array([x1, x2, x3, x4, x5, x6])
    formula_score = float(features_vec @ weights)

    return {
        "formula_score": formula_score,
        "is_formulaic": formula_score >= md["threshold"],
        "threshold": md["threshold"],
        "features": {
            "formula_direction_proj": float(x1),
            "max_template_sim": float(x2),
            "has_is_just": float(x3),
            "lacks_self_insert": float(x4),
            "cosine_to_formulaic": float(x5),
            "has_just": float(x6),
        },
        "raw_features": {
            "projection": raw_proj,
            "max_template_similarity": raw_max_sim,
            "cosine_to_formulaic_centroid": raw_cos,
        },
        "closest_template": md["template_phrases"][best_template_idx],
        "closest_template_sim": float(template_sims[best_template_idx]),
    }


# ─── MAIN ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--score":
        text = " ".join(sys.argv[2:])
        if not text:
            print("Usage: python3 formula_detector.py --score \"your opener here\"")
            sys.exit(1)
        result = score_opener(text)
        print(f"\nOpener: \"{text}\"")
        print(f"Formula Score: {result['formula_score']:.4f} "
              f"(threshold: {result['threshold']:.4f})")
        print(f"Verdict: {'FORMULAIC - REJECT' if result['is_formulaic'] else 'FRESH - ACCEPT'}")
        print(f"\nFeature breakdown:")
        for name, val in result["features"].items():
            print(f"  {name:30s}: {val:.4f}")
        print(f"\nClosest template: \"{result['closest_template']}\" "
              f"(sim={result['closest_template_sim']:.4f})")
    else:
        train_detector()
