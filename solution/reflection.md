# Reflection

Author: Lê Tuấn Anh (2A202602952)

My defense makes one metered call per event and scores each numeric metric as
a z-score against a model of clean traffic. It alerts when any metric reaches
2.5 sigma, or when a structural check fails. Structural checks cover contract
violations, a breached freshness SLA, a missing upstream edge, and an orphaned
output.

| Phase | TPR | FPR | Cost / budget | Score |
|---|---|---|---|---|
| Practice | 1.000 | 0.011 | 180 / 220 | 49.66 |
| Public | 1.000 | 0.000 | 240 / 220 | 48.18 |
| Private | 0.630 | 0.007 | within 320 | 31.28 |

I ran the private phase once and did not tune the defense after I saw its
result. The private TPR drops to 0.63 while the FPR stays near zero. This
means that the 2.5-sigma threshold is too conservative for the subtle faults
in the private stream: the misses cost much more than the false positives
that a lower threshold would add.

## Which fault types were hardest to catch, and why?

The subtle-magnitude faults were the hardest. In the public stream, subtle
corpus staleness (`avg_doc_age_days` 48.3), embedding drift
(`centroid_shift` 0.040), and a volume spike (`row_count` 557) all sit just
inside the published 3-sigma bounds. A bare threshold check on `ctx.baseline`
misses all three.

The published bounds also do not have equal quality:

- Two-sided bounds (`row_count`, `mean_amount`) give the mean and sigma
  exactly, so z-scores from them are reliable.
- One-sided bounds give only mean + 3 sigma. I estimate the mean online as the
  median of quiet events, and then derive sigma from the bound.
- The feature skew bound (0.41 sigma) is tighter than real clean noise. Clean
  batches reached 0.47 sigma. For this metric, I estimate sigma online with
  median absolute deviation and require a minimum effect of 0.5 train sigmas.

The first few events of each type were also hard. With no history, the online
estimates are not available, so the defense falls back to the bare bound. A
subtle fault that arrives in that window is missed. I reduced the warm-up
window from six samples to three for this reason.

## What would you change about your cost/coverage tradeoff, if you had another pass?

Each event exposes its signal through exactly one tool, and the payloads carry
no free signal. Because of this, full coverage costs about 1.5 credits per
event. In private, that is about 300 of 320 credits. Skipping events does not
pay: one missed fault costs about 0.5 / n_faulty of the score. One credit of
overage costs only 0.2 / budget. So I call one tool on every event, even
when I am over budget.

With another pass, I would make these changes:

- Set thresholds per event type from expected costs, not with one global
  `Z_ALERT`. A false positive costs 0.3 / n_clean and a missed fault costs
  0.5 / n_faulty. Faults are about a quarter of events, so the best threshold
  is lower than 3 sigma, but it depends on each metric's tail shape. The
  private result supports this: a FPR of 0.007 leaves room to trade a few
  false positives for many more caught faults.
- Combine weak evidence in `data_batch`. Today each metric votes alone. A
  joint score, such as the sum of squared z-scores, could catch a batch that
  is slightly off on two metrics at once.
- Seed the warm-up window with a prior from the baseline, not the bare bound,
  so that early subtle faults are not missed.
