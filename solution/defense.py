"""
Data Siege defense.

Strategy: one metered call per event (the cheapest call that exposes the
signal), then score every numeric metric as a z-score against a clean-traffic
model, and check structural facts (contract violations, lineage edges)
exactly.

The clean-traffic model comes from two sources:
  * ctx.baseline publishes bounds at mean +/- 3 sigma. For two-sided bounds
    (min/max pairs) that recovers mean and sigma exactly.
  * For one-sided bounds (only the upper limit is published) the mean is
    unknown, so it is estimated online as the median of values from events
    that did not alert. Sigma is then (bound - mean) / 3.
Feature skew is scored on the signed shift (serve - train) / train_std with
an online robust sigma (median / MAD), floored at the published bound / 3.
Its clean noise has heavier tails than the published bound implies, so it
also needs a minimum effect size before it alerts.

An event alerts when any metric exceeds Z_ALERT sigma. Z_ALERT is below the
3-sigma baseline so that subtle faults are caught, at a small and bounded
false-positive cost.
"""
from api import Verdict

Z_ALERT = 2.5             # alert threshold for single-metric z-scores
WARMUP = 3                # clean samples needed before trusting an online estimate
FEATURE_MIN_EFFECT = 0.5  # ignore feature skew below this many train sigmas
HISTORY_MAX = 200    # rolling window for online estimates


def register(ctx):
    ctx.state["hist"] = {}
    ctx.state["upstream"] = {}
    ctx.on("data_batch", _safe(check_data_batch, "checks"))
    ctx.on("contract_checkpoint", _safe(check_contract_checkpoint, "contracts"))
    ctx.on("lineage_run", _safe(check_lineage_run, "lineage"))
    ctx.on("feature_materialization", _safe(check_feature_materialization, "ai_infra"))
    ctx.on("embedding_batch", _safe(check_embedding_batch, "ai_infra"))


def _safe(handler, pillar):
    """An exception in a handler would kill the whole run. Stay quiet instead."""
    def wrapped(payload, ctx):
        try:
            return handler(payload, ctx)
        except Exception as exc:  # noqa: BLE001 - one bad event must not end the run
            return Verdict(alert=False, confidence=0.0, reason=f"handler error: {exc!r}", pillar=pillar)
    return wrapped


# ---------------------------------------------------------------- statistics

def _median(xs):
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _hist(ctx, name):
    return ctx.state["hist"].setdefault(name, [])


def _remember(ctx, pending):
    """Add metric values to the clean history. Only called for quiet events,
    so faulty values never contaminate the model."""
    for name, value in pending:
        h = _hist(ctx, name)
        h.append(value)
        if len(h) > HISTORY_MAX:
            del h[0]


def _z_two_sided(value, lo, hi):
    mu = 0.5 * (lo + hi)
    sigma = (hi - lo) / 6.0
    return abs(value - mu) / sigma if sigma > 0 else 0.0


def _z_upper(ctx, name, value, upper):
    """z-score for a metric with only a published upper bound (mean + 3 sigma)."""
    h = _hist(ctx, name)
    if len(h) < WARMUP:
        # No location estimate yet: fall back to the bare 3-sigma bound.
        return Z_ALERT if value > upper else 0.0
    mu = _median(h)
    sigma = (upper - mu) / 3.0
    if sigma <= 0:
        return 0.0
    return (value - mu) / sigma


def _z_online(ctx, name, value, sigma_floor):
    """Two-sided z-score from an online robust estimate (median / MAD)."""
    h = _hist(ctx, name)
    if len(h) < WARMUP:
        return abs(value) / sigma_floor
    mu = _median(h)
    sigma = max(1.4826 * _median([abs(x - mu) for x in h]), sigma_floor)
    return abs(value - mu) / sigma


def _verdict(ctx, pillar, scores, pending, hard_reasons=()):
    """scores: list of (name, z). Alert if any hard reason or any z >= Z_ALERT."""
    reasons = list(hard_reasons)
    worst = 0.0
    for name, z in scores:
        worst = max(worst, z)
        if z >= Z_ALERT:
            reasons.append(f"{name} z={z:.2f}")
    if reasons:
        conf = 1.0 if hard_reasons else min(1.0, worst / (2 * Z_ALERT))
        return Verdict(alert=True, confidence=conf, reason="; ".join(reasons), pillar=pillar)
    _remember(ctx, pending)
    return Verdict(alert=False, confidence=1.0 - worst / (2 * Z_ALERT), pillar=pillar)


def _error_verdict(pillar, result):
    # A tool error means we have no evidence; staying quiet protects FPR.
    return Verdict(alert=False, confidence=0.0, reason=f"tool error: {result.get('error')}", pillar=pillar)


# ------------------------------------------------------------------ handlers

def check_data_batch(payload, ctx):
    b = ctx.baseline
    r = ctx.tools.batch_profile(payload["batch_id"])
    if "error" in r:
        return _error_verdict("checks", r)

    null_rate = (r.get("null_rate") or {}).get("customer_id", 0.0)
    scores = [
        ("row_count", _z_two_sided(r["row_count"], b["row_count_min"], b["row_count_max"])),
        ("mean_amount", _z_two_sided(r["mean_amount"], b["mean_amount_min"], b["mean_amount_max"])),
        ("null_rate", _z_upper(ctx, "null_rate", null_rate, b["null_rate_max"])),
        ("staleness", _z_upper(ctx, "staleness", r["staleness_min"], b["staleness_min_max"])),
    ]
    pending = [("null_rate", null_rate), ("staleness", r["staleness_min"])]
    return _verdict(ctx, "checks", scores, pending)


def check_contract_checkpoint(payload, ctx):
    b = ctx.baseline
    r = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if "error" in r:
        return _error_verdict("contracts", r)

    hard = [f"violation: {v}" for v in r.get("violations") or []]
    delay = r["freshness_delay_min"]
    sla = (payload.get("declared_sla") or {}).get("freshness_min")
    if sla is not None and delay > sla:
        hard.append(f"freshness {delay} > declared SLA {sla}")
    scores = [("freshness_delay", _z_upper(ctx, "freshness_delay", delay, b["freshness_delay_max_min"]))]
    return _verdict(ctx, "contracts", scores, [("freshness_delay", delay)], hard)


def check_lineage_run(payload, ctx):
    b = ctx.baseline
    r = ctx.tools.lineage_graph_slice(payload["run_id"])
    if "error" in r:
        return _error_verdict("lineage", r)

    hard = []
    job = payload.get("job", "")
    actual_up = set(r.get("actual_upstream") or [])
    declared_up = {i.get("name") for i in payload.get("inputs") or []}
    missing = declared_up - actual_up
    # Learned expected upstream set per job: the union of upstreams seen on
    # quiet runs. A run that drops an edge we normally see is suspicious.
    known_up = set(ctx.state["upstream"].get(job, []))
    missing |= known_up - actual_up
    if missing:
        hard.append(f"missing upstream: {sorted(missing)}")

    n_outputs = len(payload.get("outputs") or [])
    if r.get("actual_downstream_count", 0) < max(1, n_outputs):
        hard.append(f"orphan output: downstream_count={r.get('actual_downstream_count')}")

    duration = r["duration_ms"]
    scores = [("duration", _z_upper(ctx, "duration", duration, b["lineage_duration_ms_max"]))]
    v = _verdict(ctx, "lineage", scores, [("duration", duration)], hard)
    if not v.alert:
        ctx.state["upstream"][job] = sorted(known_up | actual_up)
    return v


def check_feature_materialization(payload, ctx):
    b = ctx.baseline
    r = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if "error" in r:
        return _error_verdict("ai_infra", r)

    if r.get("train_std"):
        shift = (r["serve_mean"] - r["train_mean"]) / r["train_std"]
    else:
        shift = r["mean_shift_sigma"]
    z = _z_online(ctx, "feature_shift", shift, b["feature_mean_shift_sigma_max"] / 3.0)
    if abs(shift) < FEATURE_MIN_EFFECT:
        z = min(z, Z_ALERT - 0.01)
    scores = [("feature_shift", z)]
    return _verdict(ctx, "ai_infra", scores, [("feature_shift", shift)])


def check_embedding_batch(payload, ctx):
    b = ctx.baseline
    r = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if "error" in r:
        return _error_verdict("ai_infra", r)

    shift = r["centroid_shift"]
    age = r["avg_doc_age_days"]
    scores = [
        ("centroid_shift", _z_upper(ctx, "centroid_shift", shift, b["embedding_centroid_shift_max"])),
        ("doc_age", _z_upper(ctx, "doc_age", age, b["corpus_avg_doc_age_days_max"])),
    ]
    return _verdict(ctx, "ai_infra", scores, [("centroid_shift", shift), ("doc_age", age)])
