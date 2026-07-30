# Upstream PR evaluation — 2026-08-02

Scope: `Acters/next-plaid` is a personal fork for the dreb ColGREP runtime. Upstream PRs are evaluation inputs only. PR 166 remains untouched; PRs 167 and 168 remain closed. The new integration lineage is `dreb-colgrep-runtime`, created from upstream release `1.6.5` (`76092e1`).

## PR 162 — adaptive cold indexing

Disposition: **adapt selectively; do not cherry-pick wholesale**.

Useful pieces:

- bounded ONNX worker pool
- per-document token-limit plumbing
- potential one-write fresh indexing after interruption/memory validation
- parser fixes only after independent evaluation

Blocking concerns:

- adaptive embedding policy is not persisted in index identity/state, so reruns can mix policies silently
- reported semantic-only retrieval regressed materially on one private workload
- fresh-build one-write behavior may increase retained embeddings and lose resumability
- extreme document-length overrides are not safely validated
- parser/Flow changes need separate coverage

Fork decision: keep adaptive context opt-in. Persist any effective indexing policy or require an explicit forced rebuild when it changes. Validate dreb retrieval before considering it as a default.

## PR 169 — asymmetric residual LUT scoring

Disposition: **adapt as opt-in; preserve float scoring by default**.

This is not the previously rejected query-only-INT8 experiment: centroid scoring and inverse-norm correction remain exact FP32, while INT8 is limited to residual correction. However, it is not mathematically identical to float scoring and the upstream NDCG evidence does not establish MRR safety for the dreb workload.

Blocking concerns before enablement:

- measure MRR@10, NDCG@10, Recall@k, and top-k overlap on the fork workload
- first asym search computes inverse norms for every token and adds process-local memory
- dense search temporarily holds original and transposed centroid matrices
- persistent-server first-request latency and parallel/custom-pool behavior need tests
- non-finite score behavior and dimension/codec fallbacks need coverage

Fork decision: integrate only behind an explicit opt-in after quality and memory gates pass. It is not required for the initial replacement rollout.

## PR 170 — stage-1 shortlist pipeline

Disposition: **defer for adaptation; do not cherry-pick while stacked on PR 169**.

Blocking concerns:

- commits depend on PR 169's caches/scoring interfaces
- `n_full_scores = 0` behavior regresses from no results to scoring top-k
- stage-1 global quantization can reorder sums across query tokens, even when per-token maxima are monotonic
- NaN/infinity comparator semantics differ from existing selection
- lazy per-token caches add substantial process-local memory even for some non-asym paths
- existing tests measure score error, not shortlist/top-k equivalence

Fork decision: defer until PR 169 has passed quality/memory gates. Then adapt stage-1 separately with shortlist-ID, recall, top-k overlap, adversarial tie/non-finite, and bounded-memory tests.

## Immediate integration order

1. Start from upstream `76092e1` on `dreb-colgrep-runtime`.
2. Integrate and harden the stdio server/read-only-loading work without changing PR-166's branch.
3. Integrate exact overlap.
4. Integrate validated model-cache resolution.
5. Integrate bounded codec/pooling controls.
6. Add local fixes for genuine blockers from the PR-166 audit.
7. Revisit PRs 162/169/170 only after the replacement runtime is stable.
