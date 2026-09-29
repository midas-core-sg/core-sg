# Investigate first-run NNDescent latency in `score_sg_script`

## Context

During experiments based on `benchmarking/run_time/scripts/score_sg_script.py`, we observed that the first `NNDescent` execution has noticeably higher latency than subsequent executions.

This behavior still appears even when the call happens inside the repetition loop, which suggests there may be some initialization, warm-up, or internal setup cost that is not being isolated by the current benchmark.

## Observed behavior

- The first repetition of `score.fit(X, k_max=k_max)` tends to be slower.
- Later repetitions are more stable and usually faster.
- The effect was observed specifically in the `PyNNDescent` path, through `NNDescent`, inside `core_sg/score_sg.py`.

## Investigation target

The approximate neighbor graph is currently built in:

- `core_sg/score_sg.py`
- `build_approximate_knn_graph(...)`
- the line where `index = NNDescent(X, metric=_resolve_metric(metric), **kwargs)` is executed

In the benchmark, this is triggered by:

- `benchmarking/run_time/scripts/score_sg_script.py`
- `run_score_sg_variant(...)`
- the `score.fit(X, k_max=k_max)` call inside the repetition loop

## Initial hypotheses

- JIT compilation or warm-up cost in dependencies used by `PyNNDescent`
- internal data-structure initialization during the first index build
- memory/cache warm-up, allocations, or dataset preprocessing effects
- side effects from numerical backend initialization in underlying dependencies

## External references

- PyNNDescent documentation states that the first query can be slow because of internal bookkeeping and because Numba may JIT-compile many routines in the background. The same page also notes that the very first index build may take longer than expected, while subsequent runs are faster:
  https://pynndescent.readthedocs.io/en/stable/how_to_use_pynndescent.html
- PyNNDescent documentation also explains that query preparation includes extra preprocessing work on the graph, and that this work is intentionally deferred unless query-time behavior is needed:
  https://pynndescent.readthedocs.io/en/stable/how_pynndescent_works.html
- Numba documentation confirms that, in lazy compilation mode, compilation is deferred until the first function execution:
  https://numba.readthedocs.io/en/stable/user/jit.html
- Numba developer documentation describes caching compiled functions to reduce future compilation overhead when cacheable:
  https://numba.readthedocs.io/en/latest/developer/caching.html

## Working interpretation

The most plausible explanation is that the first-run slowdown is largely caused by one-time JIT compilation and runtime warm-up in the `PyNNDescent`/Numba stack, possibly combined with one-time internal initialization during index construction.

In our specific case, `score-sg` reads `NNDescent.neighbor_graph` directly and does not call `query(...)`, so the likely explanation is less about query preparation and more about first-use compilation and initialization during index construction itself.

## Goal

Understand why the first `NNDescent` execution is slower and decide how this should be handled in benchmarks and, if relevant, in the `score-sg` workflow.

## Proposed tasks

- Reproduce the behavior with logging and timing separated between the first repetition and the remaining ones.
- Measure the isolated `NNDescent` construction time versus later stages of `score.fit(...)`.
- Check whether the effect persists across different processes or only happens once per process.
- Test whether an explicit warm-up before benchmarking reduces the discrepancy.
- Validate whether the behavior depends on dataset size, `k_max`, metric, or `random_state`.
- Review `PyNNDescent` documentation and issue trackers to confirm whether this warm-up cost is known.

## Exit criteria

- A plausible and reproducible explanation for the first-run latency.
- A clear recommendation for benchmarking:
- keep the current approach,
- discard the first repetition,
- add an explicit warm-up step,
- or report cold start and steady-state timings separately.
- If needed, update `score_sg_script.py` so the effect can be measured explicitly.
