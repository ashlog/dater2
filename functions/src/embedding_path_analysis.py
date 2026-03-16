"""
Embedding Path Analysis for Hinge Openers
==========================================
Analyzes word-by-word embedding traversal patterns to find correlations
between semantic path characteristics and opener quality scores.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
EVAL_FILE = "/Users/ash/projects/dater2/functions/src/eval_results_gen_2026-03-16T02-22-00.json"
SCORE_DIMS = [
    "reply_likelihood", "humor_quality", "creativity", "specificity",
    "human_likeness", "closer_strength", "warmth_signal", "formula_freshness"
]
PATH_METRICS = [
    "avg_step_size", "max_jump", "path_variance", "total_path_length",
    "direction_changes", "final_jump", "opening_jump"
]

# ── Load data ───────────────────────────────────────────────────────────────
print("Loading eval results...")
with open(EVAL_FILE, "r") as f:
    data = json.load(f)

results = data["results"]
print(f"  Loaded {len(results)} openers\n")

# ── Load model ──────────────────────────────────────────────────────────────
print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("  Model loaded.\n")

# ── Helper functions ────────────────────────────────────────────────────────

def tokenize_to_words(text: str) -> list[str]:
    """Simple whitespace + punctuation-aware tokenization."""
    # Split on whitespace, keep punctuation attached to words
    return text.split()


def compute_embedding_path(words: list[str], model) -> list[float]:
    """Embed each word individually and compute cosine distances between consecutive pairs."""
    if len(words) < 2:
        return []
    embeddings = model.encode(words, show_progress_bar=False)
    distances = []
    for i in range(len(embeddings) - 1):
        d = cosine_dist(embeddings[i], embeddings[i + 1])
        distances.append(float(d))
    return distances


def compute_ngram_path(words: list[str], n: int, model) -> list[float]:
    """Embed n-grams (overlapping) and compute cosine distances between consecutive n-gram embeddings."""
    if len(words) < n + 1:
        return []
    ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
    embeddings = model.encode(ngrams, show_progress_bar=False)
    distances = []
    for i in range(len(embeddings) - 1):
        d = cosine_dist(embeddings[i], embeddings[i + 1])
        distances.append(float(d))
    return distances


def path_metrics(distances: list[float]) -> dict:
    """Compute all path metrics from a list of step distances."""
    if not distances:
        return {m: None for m in PATH_METRICS}

    arr = np.array(distances)

    # Direction changes: count sign changes in the first derivative of step sizes
    direction_changes = 0
    if len(arr) > 1:
        diffs = np.diff(arr)
        signs = np.sign(diffs)
        signs = signs[signs != 0]  # ignore zero changes
        if len(signs) > 1:
            direction_changes = int(np.sum(np.abs(np.diff(signs)) > 0))

    return {
        "avg_step_size": float(np.mean(arr)),
        "max_jump": float(np.max(arr)),
        "path_variance": float(np.var(arr)),
        "total_path_length": float(np.sum(arr)),
        "direction_changes": direction_changes,
        "final_jump": float(arr[-1]),
        "opening_jump": float(arr[0]),
    }


def extract_scores(result: dict) -> dict:
    """Extract numeric scores from a result entry."""
    return {dim: result["scores"][dim]["score"] for dim in SCORE_DIMS}


# ── Main analysis ───────────────────────────────────────────────────────────

print("=" * 80)
print("EMBEDDING PATH ANALYSIS — WORD-BY-WORD TRAVERSAL PATTERNS")
print("=" * 80)

# Step 1: Compute paths and metrics for all openers
print("\nComputing embedding paths for all openers...")
records = []
for i, r in enumerate(results):
    comment = r["comment"]
    words = tokenize_to_words(comment)
    scores = extract_scores(r)
    avg_score = np.mean(list(scores.values()))

    # 1-gram path (word-by-word)
    distances_1g = compute_embedding_path(words, model)
    metrics_1g = path_metrics(distances_1g)

    # 2-gram path
    distances_2g = compute_ngram_path(words, 2, model)
    metrics_2g = path_metrics(distances_2g)

    # 3-gram path
    distances_3g = compute_ngram_path(words, 3, model)
    metrics_3g = path_metrics(distances_3g)

    records.append({
        "comment": comment,
        "word_count": len(words),
        "scores": scores,
        "avg_score": avg_score,
        "metrics_1g": metrics_1g,
        "metrics_2g": metrics_2g,
        "metrics_3g": metrics_3g,
        "distances_1g": distances_1g,
        "model_variant": r["model"],
    })

    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/{len(results)}...")

print(f"  Done. {len(records)} openers processed.\n")

# Filter out records with None metrics (too short)
valid_1g = [r for r in records if r["metrics_1g"]["avg_step_size"] is not None]
valid_2g = [r for r in records if r["metrics_2g"]["avg_step_size"] is not None]
valid_3g = [r for r in records if r["metrics_3g"]["avg_step_size"] is not None]
print(f"  Valid for 1-gram analysis: {len(valid_1g)}")
print(f"  Valid for 2-gram analysis: {len(valid_2g)}")
print(f"  Valid for 3-gram analysis: {len(valid_3g)}")

# ── Section 1: Descriptive statistics ───────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 1: DESCRIPTIVE STATISTICS (1-gram paths)")
print("=" * 80)

for metric in PATH_METRICS:
    vals = [r["metrics_1g"][metric] for r in valid_1g]
    print(f"\n  {metric}:")
    print(f"    mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
          f"min={np.min(vals):.4f}  max={np.max(vals):.4f}  "
          f"median={np.median(vals):.4f}")

# ── Section 2: Spearman correlations ───────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 2: SPEARMAN CORRELATIONS — 1-gram path metrics vs scores")
print("=" * 80)

def correlation_table(valid_records, gram_key, label):
    print(f"\n{'':>20s}", end="")
    for dim in SCORE_DIMS:
        print(f"{dim:>16s}", end="")
    print(f"{'avg_score':>16s}")
    print("-" * (20 + 17 * (len(SCORE_DIMS) + 1)))

    significant_findings = []

    for metric in PATH_METRICS:
        metric_vals = [r[gram_key][metric] for r in valid_records]
        print(f"{metric:>20s}", end="")

        for dim in SCORE_DIMS + ["avg_score"]:
            if dim == "avg_score":
                score_vals = [r["avg_score"] for r in valid_records]
            else:
                score_vals = [r["scores"][dim] for r in valid_records]

            rho, p = spearmanr(metric_vals, score_vals)
            marker = "**" if p < 0.01 else "*" if p < 0.05 else "~" if p < 0.1 else ""
            print(f"{rho:>12.3f}{marker:>4s}", end="")

            if p < 0.05:
                significant_findings.append((metric, dim, rho, p))
        print()

    return significant_findings

print("\n── 1-gram (word-by-word) ──")
sig_1g = correlation_table(valid_1g, "metrics_1g", "1-gram")

print("\n── 2-gram (bigram) ──")
sig_2g = correlation_table(valid_2g, "metrics_2g", "2-gram")

print("\n── 3-gram (trigram) ──")
sig_3g = correlation_table(valid_3g, "metrics_3g", "3-gram")

# ── Section 3: Significant findings summary ─────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 3: STATISTICALLY SIGNIFICANT FINDINGS (p < 0.05)")
print("=" * 80)

for label, findings in [("1-gram", sig_1g), ("2-gram", sig_2g), ("3-gram", sig_3g)]:
    print(f"\n  {label} paths:")
    if not findings:
        print("    (none)")
    else:
        for metric, dim, rho, p in sorted(findings, key=lambda x: abs(x[2]), reverse=True):
            direction = "+" if rho > 0 else "-"
            print(f"    {direction} {metric} vs {dim}: rho={rho:.3f}, p={p:.4f}")

# ── Section 4: Top 20% vs Bottom 20% comparison ────────────────────────────
print("\n" + "=" * 80)
print("SECTION 4: TOP 20% vs BOTTOM 20% BY AVERAGE SCORE")
print("=" * 80)

sorted_records = sorted(valid_1g, key=lambda r: r["avg_score"], reverse=True)
n = len(sorted_records)
top_n = max(1, n // 5)
bot_n = max(1, n // 5)
top_20 = sorted_records[:top_n]
bot_20 = sorted_records[-bot_n:]

print(f"\n  Top 20% ({len(top_20)} openers, avg_score >= {top_20[-1]['avg_score']:.1f}):")
print(f"  Bottom 20% ({len(bot_20)} openers, avg_score <= {bot_20[0]['avg_score']:.1f}):")

print(f"\n{'Metric':>22s}  {'Top 20% mean':>14s}  {'Bot 20% mean':>14s}  {'Difference':>12s}  {'Effect':>8s}")
print("-" * 78)

for metric in PATH_METRICS:
    top_vals = [r["metrics_1g"][metric] for r in top_20]
    bot_vals = [r["metrics_1g"][metric] for r in bot_20]
    top_mean = np.mean(top_vals)
    bot_mean = np.mean(bot_vals)
    diff = top_mean - bot_mean
    # Cohen's d (rough)
    pooled_std = np.sqrt((np.std(top_vals)**2 + np.std(bot_vals)**2) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else 0
    print(f"{metric:>22s}  {top_mean:>14.4f}  {bot_mean:>14.4f}  {diff:>+12.4f}  {cohens_d:>+8.2f}d")

# Show some example openers from each group
print("\n  Top 20% examples:")
for r in top_20[:5]:
    print(f"    [{r['avg_score']:.1f}] \"{r['comment']}\"")
    print(f"         avg_step={r['metrics_1g']['avg_step_size']:.3f}  max_jump={r['metrics_1g']['max_jump']:.3f}  "
          f"final={r['metrics_1g']['final_jump']:.3f}  var={r['metrics_1g']['path_variance']:.4f}")

print("\n  Bottom 20% examples:")
for r in bot_20[:5]:
    print(f"    [{r['avg_score']:.1f}] \"{r['comment']}\"")
    print(f"         avg_step={r['metrics_1g']['avg_step_size']:.3f}  max_jump={r['metrics_1g']['max_jump']:.3f}  "
          f"final={r['metrics_1g']['final_jump']:.3f}  var={r['metrics_1g']['path_variance']:.4f}")

# ── Section 5: Semantic surprise analysis ───────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 5: SEMANTIC SURPRISE ANALYSIS")
print("=" * 80)
print("  Does having large jumps (semantic surprise) correlate with creativity/humor?")

# Define surprise as having at least one jump > 2 standard deviations above mean step size
for r in valid_1g:
    dists = r["distances_1g"]
    if len(dists) >= 2:
        mean_d = np.mean(dists)
        std_d = np.std(dists)
        r["n_surprises"] = int(np.sum(np.array(dists) > mean_d + 1.5 * std_d)) if std_d > 0 else 0
        r["surprise_ratio"] = r["n_surprises"] / len(dists)
        # Also: ratio of max jump to average
        r["jump_ratio"] = max(dists) / mean_d if mean_d > 0 else 0
    else:
        r["n_surprises"] = 0
        r["surprise_ratio"] = 0
        r["jump_ratio"] = 0

surprise_metrics = ["n_surprises", "surprise_ratio", "jump_ratio"]
print(f"\n{'':>20s}", end="")
for dim in SCORE_DIMS:
    print(f"{dim:>16s}", end="")
print(f"{'avg_score':>16s}")
print("-" * (20 + 17 * (len(SCORE_DIMS) + 1)))

for sm in surprise_metrics:
    print(f"{sm:>20s}", end="")
    for dim in SCORE_DIMS + ["avg_score"]:
        if dim == "avg_score":
            score_vals = [r["avg_score"] for r in valid_1g]
        else:
            score_vals = [r["scores"][dim] for r in valid_1g]
        metric_vals = [r[sm] for r in valid_1g]
        rho, p = spearmanr(metric_vals, score_vals)
        marker = "**" if p < 0.01 else "*" if p < 0.05 else "~" if p < 0.1 else ""
        print(f"{rho:>12.3f}{marker:>4s}", end="")
    print()

# Find examples of high-surprise openers
high_surprise = sorted(valid_1g, key=lambda r: r["jump_ratio"], reverse=True)[:10]
print("\n  Top 10 highest semantic surprise (jump_ratio) openers:")
for r in high_surprise:
    print(f"    [avg={r['avg_score']:.1f} cre={r['scores']['creativity']} hum={r['scores']['humor_quality']}] "
          f"jump_ratio={r['jump_ratio']:.2f} \"{r['comment']}\"")

# ── Section 6: Final jump and closer_strength ───────────────────────────────
print("\n" + "=" * 80)
print("SECTION 6: FINAL JUMP vs CLOSER STRENGTH (deep dive)")
print("=" * 80)

final_jumps = [r["metrics_1g"]["final_jump"] for r in valid_1g]
closer_scores = [r["scores"]["closer_strength"] for r in valid_1g]
rho, p = spearmanr(final_jumps, closer_scores)
print(f"\n  Spearman correlation: rho={rho:.3f}, p={p:.4f}")
print(f"  Interpretation: {'Significant' if p < 0.05 else 'Not significant'} at p<0.05")

# Bin by final_jump terciles
sorted_by_fj = sorted(valid_1g, key=lambda r: r["metrics_1g"]["final_jump"])
t = len(sorted_by_fj) // 3
terciles = [sorted_by_fj[:t], sorted_by_fj[t:2*t], sorted_by_fj[2*t:]]
labels = ["Low final_jump", "Mid final_jump", "High final_jump"]

print(f"\n  {'Tercile':>18s}  {'n':>4s}  {'avg final_jump':>16s}  {'avg closer_str':>16s}  {'avg creativity':>16s}  {'avg avg_score':>14s}")
print("-" * 95)
for label, group in zip(labels, terciles):
    fj = np.mean([r["metrics_1g"]["final_jump"] for r in group])
    cs = np.mean([r["scores"]["closer_strength"] for r in group])
    cr = np.mean([r["scores"]["creativity"] for r in group])
    av = np.mean([r["avg_score"] for r in group])
    print(f"  {label:>18s}  {len(group):>4d}  {fj:>16.4f}  {cs:>16.2f}  {cr:>16.2f}  {av:>14.2f}")

# Show best closers with their final jumps
print("\n  Highest closer_strength openers with their final jump:")
top_closers = sorted(valid_1g, key=lambda r: r["scores"]["closer_strength"], reverse=True)[:8]
for r in top_closers:
    words = tokenize_to_words(r["comment"])
    last_two = " | ".join(words[-2:]) if len(words) >= 2 else r["comment"]
    print(f"    [closer={r['scores']['closer_strength']}  fj={r['metrics_1g']['final_jump']:.3f}]  "
          f"\"{r['comment']}\"")
    print(f"      Last transition: \"{last_two}\"")

# ── Section 7: Opening jump analysis ────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 7: OPENING JUMP ANALYSIS")
print("=" * 80)
print("  Does a surprising first word transition hook attention?")

opening_jumps = [r["metrics_1g"]["opening_jump"] for r in valid_1g]
for dim in ["reply_likelihood", "humor_quality", "creativity", "avg_score"]:
    if dim == "avg_score":
        score_vals = [r["avg_score"] for r in valid_1g]
    else:
        score_vals = [r["scores"][dim] for r in valid_1g]
    rho, p = spearmanr(opening_jumps, score_vals)
    sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  opening_jump vs {dim}: rho={rho:.3f}, p={p:.4f} {sig}")

# ── Section 8: N-gram comparison ────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 8: 1-GRAM vs 2-GRAM vs 3-GRAM PATH COMPARISON")
print("=" * 80)
print("  Do phrase-level paths reveal different patterns than word-level?")

# For each gram size, find the strongest correlation with avg_score
for gram, valid, key in [(1, valid_1g, "metrics_1g"), (2, valid_2g, "metrics_2g"), (3, valid_3g, "metrics_3g")]:
    print(f"\n  {gram}-gram strongest correlations with avg_score (n={len(valid)}):")
    score_vals = [r["avg_score"] for r in valid]

    corrs = []
    for metric in PATH_METRICS:
        metric_vals = [r[key][metric] for r in valid]
        rho, p = spearmanr(metric_vals, score_vals)
        corrs.append((metric, rho, p))

    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for metric, rho, p in corrs[:5]:
        sig = "**" if p < 0.01 else "*" if p < 0.05 else "~" if p < 0.1 else ""
        print(f"    {metric:>20s}: rho={rho:+.3f}, p={p:.4f} {sig}")

# ── Section 9: Word count as confounder check ──────────────────────────────
print("\n" + "=" * 80)
print("SECTION 9: WORD COUNT CONFOUNDER CHECK")
print("=" * 80)

word_counts = [r["word_count"] for r in valid_1g]
avg_scores = [r["avg_score"] for r in valid_1g]
rho_wc, p_wc = spearmanr(word_counts, avg_scores)
print(f"\n  word_count vs avg_score: rho={rho_wc:.3f}, p={p_wc:.4f}")

# Check word count vs path metrics
for metric in PATH_METRICS:
    metric_vals = [r["metrics_1g"][metric] for r in valid_1g]
    rho, p = spearmanr(word_counts, metric_vals)
    sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  word_count vs {metric}: rho={rho:.3f}, p={p:.4f} {sig}")

# Partial correlation: control for word count
print("\n  Note: total_path_length will naturally correlate with word_count.")
print("  avg_step_size and path_variance are length-normalized and more informative.")

# ── Section 10: Per-variant analysis ────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 10: PATH METRICS BY PROMPT VARIANT")
print("=" * 80)

variants = set(r["model_variant"] for r in valid_1g)
for variant in sorted(variants):
    subset = [r for r in valid_1g if r["model_variant"] == variant]
    print(f"\n  {variant} (n={len(subset)}):")
    for metric in PATH_METRICS:
        vals = [r["metrics_1g"][metric] for r in subset]
        print(f"    {metric:>20s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")
    avg = np.mean([r["avg_score"] for r in subset])
    print(f"    {'avg_score':>20s}: mean={avg:.2f}")

# ── ACTIONABLE INSIGHTS ────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("ACTIONABLE INSIGHTS SUMMARY")
print("=" * 80)

# Compute key findings for the summary
print("""
This section synthesizes the findings into concrete recommendations.
""")

# 1. Find the metric most correlated with avg_score
all_corrs_1g = []
for metric in PATH_METRICS:
    metric_vals = [r["metrics_1g"][metric] for r in valid_1g]
    rho, p = spearmanr(metric_vals, [r["avg_score"] for r in valid_1g])
    all_corrs_1g.append((metric, rho, p))
all_corrs_1g.sort(key=lambda x: abs(x[1]), reverse=True)

print("  TOP CORRELATIONS WITH OVERALL QUALITY:")
for metric, rho, p in all_corrs_1g:
    sig = " (SIGNIFICANT)" if p < 0.05 else " (trending)" if p < 0.1 else ""
    direction = "Higher" if rho > 0 else "Lower"
    print(f"    {direction} {metric} → better openers (rho={rho:+.3f}, p={p:.4f}){sig}")

# 2. Compute the "ideal" path profile from top performers
print("\n  IDEAL PATH PROFILE (from top 20% openers):")
for metric in PATH_METRICS:
    top_vals = [r["metrics_1g"][metric] for r in top_20]
    print(f"    {metric:>20s}: {np.mean(top_vals):.4f} +/- {np.std(top_vals):.4f}")

# 3. Key takeaways
print("""
  KEY TAKEAWAYS:

  1. PATH VARIANCE: High path variance means erratic semantic jumps — check if
     this helps or hurts. If positively correlated with creativity/humor, it
     suggests "semantic zigzagging" is a feature, not a bug.

  2. FINAL JUMP: If final_jump positively correlates with closer_strength,
     this validates the "surprise closer" pattern — end with a semantically
     unexpected word/phrase.

  3. OPENING JUMP: If opening_jump correlates with reply_likelihood, strong
     semantic hooks at the start matter.

  4. AVG STEP SIZE: This measures overall semantic diversity per transition.
     Very high = disjointed; very low = repetitive. Look for the sweet spot.

  5. N-GRAM DIFFERENCES: If 2-gram or 3-gram paths show stronger correlations
     than 1-gram, it means phrase-level coherence matters more than word-level
     jumps — openers should have coherent chunks that contrast with each other.
""")

# 4. Practical filter thresholds
print("  SUGGESTED QUALITY FILTER THRESHOLDS (from top 20%):")
for metric in ["avg_step_size", "path_variance", "final_jump", "max_jump"]:
    top_vals = [r["metrics_1g"][metric] for r in top_20]
    bot_vals = [r["metrics_1g"][metric] for r in bot_20]
    t_mean = np.mean(top_vals)
    b_mean = np.mean(bot_vals)
    if t_mean > b_mean:
        threshold = np.percentile(top_vals, 25)  # bottom of good range
        print(f"    {metric}: prefer >= {threshold:.4f} (top 20% P25)")
    else:
        threshold = np.percentile(top_vals, 75)  # top of good range
        print(f"    {metric}: prefer <= {threshold:.4f} (top 20% P75)")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
