#!/usr/bin/env python3
"""Perf harness for the view-assign fix's Python-side bookkeeping.

Run directly on any branch and compare numbers:  PYTHONPATH=. python3 extra/view_assign_stress_perf.py

What it measures and why (all timings are host-side graph bookkeeping, not GPU work):

  - view assign cost vs the number/depth of LIVE tensors: the fix scans every live tensor and
    toposorts its graph on EVERY view assign (Tensor.assign reader_graphs). Cost is therefore
    O(live tensors x graph depth) per assign even when nothing reads the target buffer.
    Review measurement 2026-08-21 (M-series mac, 200 live buffers with depth-20 graphs):
        base (pre-fix):  ~2.9 ms per view assign
        fix branch:     ~108  ms per view assign   (~38x)
    Un-jitted view-assign loops (e.g. KV-cache decode without TinyJit, or the capture pass of a
    big model) pay this. Jitted replay does not (python assign only runs during capture).

  - the eager snapshot: a view assign with an overlapping unrealized prior reader launches a
    whole-buffer copy kernel INSIDE assign() (laziness break, one copy per such assign).

  - guard rails: simple (non-view) assigns and view assigns with no prior readers must stay
    lazy (0 kernels inside assign) and realize cost must not blow up.
"""
import time
from tinygrad import Tensor
from tinygrad.helpers import GlobalCounters

def bench(label, fn, warn_ms=None):
  t0 = time.perf_counter()
  fn()
  dt = (time.perf_counter() - t0) * 1000
  flag = "  <-- SLOW" if warn_ms is not None and dt > warn_ms else ""
  print(f"  {label:58s} {dt:9.1f} ms{flag}")
  return dt

def population(n_bufs, chain):
  pop = []
  for _ in range(n_bufs):
    t = Tensor.ones(16).contiguous().realize()
    for _ in range(chain): t = t + 1
    pop.append(t)
  return pop

print("view assign scaling vs live-tensor population (100 assigns each, no readers of target)")
per_assign = {}
for n_bufs, chain in [(0, 0), (50, 20), (200, 20), (200, 100)]:
  pop = population(n_bufs, chain)
  target = Tensor.ones(64).contiguous().realize()
  def assigns():
    for i in range(100): target[i % 63:i % 63 + 1].assign(Tensor([1.0]))
  dt = bench(f"population {n_bufs:4d} bufs x depth {chain:3d}", assigns)
  per_assign[(n_bufs, chain)] = dt / 100
  del pop, target
print(f"  per-assign: {'  '.join(f'{k}={v:.2f}ms' for k, v in per_assign.items())}")

print("hot-path guard rails")
target = Tensor.ones(1024).contiguous().realize()
GlobalCounters.reset()
target[:1].assign(Tensor([9.0]))
k = GlobalCounters.kernel_count
print(f"  kernels inside view assign, no readers                    {k:9d}    (want 0: lazy)")
assert k == 0, "view assign without readers is no longer lazy!"

target2 = Tensor.ones(1024).contiguous().realize()
readers = [target2 + i for i in range(4)]
GlobalCounters.reset()
target2[:1].assign(Tensor([9.0]))
k = GlobalCounters.kernel_count
print(f"  kernels inside view assign, 4 overlapping prior readers   {k:9d}    (eager snapshot copy)")

simple = [Tensor.ones(16).contiguous().realize() for _ in range(200)]
bench("200 simple assigns (optimizer-style)", lambda: [t.assign(t + 1) for t in simple], warn_ms=100)
bench("realize the 200 assigned tensors", lambda: Tensor.realize(*simple))

fresh = [Tensor.ones(16).contiguous() for _ in range(200)]
bench("realize 200 fresh tensors (no assigns anywhere)", lambda: Tensor.realize(*fresh))

print("done - compare these numbers against the pre-fix base commit")
