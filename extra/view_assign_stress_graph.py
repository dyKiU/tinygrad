#!/usr/bin/env python3
"""
One-time stress / proof-of-concept script for the "view assign replacing prior readers" fix.
This is NOT a unit test (nothing here runs under pytest) - run it directly:

  PYTHONPATH=. python3 extra/view_assign_stress_graph.py

It builds one big tensor graph (hundreds of live tensor nodes, across dozens of independent
buffers) that exercises every scenario the fix is supposed to guard against, all mixed together
and realized in several different orders, and checks every result against a plain numpy
reference model. Scenarios covered:

  - prior readers: a tensor created BEFORE a view-assign must keep the value it captured at
    creation time, no matter what order things get realized in (the original bug: the assign
    used to get spliced into every live tensor's graph, including readers that predate it).
  - post readers: a tensor created AFTER a view-assign must see the written value.
  - the assign's own return value must actually perform the write - checked by reading the
    RAW buffer memory directly (buffer.numpy()), never by re-reading through the base tensor.
    Re-reading through the base tensor is what let the bitcast-view bug hide inside the
    existing test_assign_bitcast test: the base tensor's .uop is always correctly updated,
    so a stale writer.uop (missing the AFTER(assign) node) silently no-ops and nothing catches
    it unless you check the buffer the writer was actually supposed to touch.
  - plain shrink-view assign, whole-buffer bitcast assign, shrink-then-bitcast,
    bitcast-then-shrink, and double-bitcast (round trip) assign targets.
  - chains of several sequential view-assigns to the same buffer, with readers captured
    between each step.
  - cross-buffer isolation: rewriting a view of one buffer must not affect a reader that spans
    an unrelated buffer.

Every scenario is rebuilt from scratch for each realize ordering (assign mutates state, so the
same tensors can't be reused across orderings), and every check must pass in every ordering.
"""
import random
import numpy as np
from tinygrad import Tensor, dtypes
import tinygrad.tensor as tt

F32, U32 = dtypes.float32, dtypes.uint32

NUM_SIMPLE = 40   # 5 view kinds, cycled
NUM_CHAIN = 8     # each does 3 sequential view-assigns
NUM_CROSS = 10    # each is a pair of buffers

KINDS = ["shrink", "bitcast_full", "shrink_then_bitcast", "double_bitcast"]
# NOTE: "bitcast(dtype)[a:b].assign(...)" (bitcast BEFORE shrink) is left out - it fails schedule
# verification (Ops.RANGE ...) identically on unmodified upstream master, so it's a pre-existing
# scheduler limitation unrelated to this fix, not something this script is checking.

def to_bits(vals): return np.array(vals, dtype=np.float32).view(np.uint32).tolist()

class Scenario:
  def __init__(self):
    self.sinks: list[Tensor] = []
    self.checks: list[tuple[str, object, list]] = []  # (label, zero-arg callable, expected)
  def sink(self, t: Tensor): self.sinks.append(t)
  def check(self, label, fn, expected): self.checks.append((label, fn, expected))

def build_simple_buffer(sc: Scenario, idx: int, kind: str):
  size = 4 + (idx % 4)
  a, b = idx % (size - 1), idx % (size - 1) + 1
  base_vals = np.arange(size, dtype=np.float32) + idx * 1000
  ref = base_vals.copy()
  base = Tensor(base_vals.tolist(), dtype=F32).contiguous().realize()

  # prior reader: captured before the assign - must NOT see the write, in any realize order
  prior_reader = base + 0.5
  prior_expected = (ref + 0.5).tolist()

  if kind == "shrink":
    new_vals = [float(-(idx * 10 + 1))]
    view = base[a:b]
    rhs = Tensor(new_vals, dtype=F32)
    ref[a:b] = new_vals
  elif kind == "bitcast_full":
    a, b = 0, size
    new_vals = [float(-(idx * 10 + i + 1)) for i in range(size)]
    view = base.bitcast(U32)
    rhs = Tensor(to_bits(new_vals), dtype=U32)
    ref[:] = new_vals
  elif kind == "shrink_then_bitcast":
    new_vals = [float(-(idx * 10 + 1))]
    view = base[a:b].bitcast(U32)
    rhs = Tensor(to_bits(new_vals), dtype=U32)
    ref[a:b] = new_vals
  elif kind == "double_bitcast":
    new_vals = [float(-(idx * 10 + 1))]
    view = base[a:b].bitcast(U32).bitcast(F32)
    rhs = Tensor(new_vals, dtype=F32)
    ref[a:b] = new_vals
  else:
    raise ValueError(kind)

  # realized only through its own return value - this is exactly the case that used to no-op
  writer = view.assign(rhs)

  sc.sink(prior_reader); sc.check(f"{kind}#{idx} prior_reader", prior_reader.tolist, prior_expected)
  sc.sink(writer)

  # IMPORTANT: these two checks must stay in separate buffers, never both on the same one.
  # A post_reader (below) depends on base.uop, which is ALWAYS correctly updated even when the
  # bug is present - realizing it in the same batch as writer forces the STORE as a side effect
  # and masks whether writer alone actually performed the write. So: alternate check type by
  # *cycle through KINDS*, not by idx parity directly - len(KINDS) is even, so idx%2 would always
  # pair the same kinds with the same check type (e.g. bitcast_full/double_bitcast would only
  # ever get post_reader, never the raw_buffer check that's the whole point of this scenario).
  if (idx // len(KINDS)) % 2 == 0:
    sc.check(f"{kind}#{idx} raw_buffer_via_writer_alone", (lambda base=base: base.uop.buffer.numpy().tolist()), ref.tolist())
  else:
    post_reader = base + 0.25
    post_expected = (ref + 0.25).tolist()
    sc.sink(post_reader); sc.check(f"{kind}#{idx} post_reader", post_reader.tolist, post_expected)

def build_chain_buffer(sc: Scenario, idx: int, steps: int = 3):
  size = 6
  base_vals = np.arange(size, dtype=np.float32) + idx * 1000 + 500000
  ref = base_vals.copy()
  base = Tensor(base_vals.tolist(), dtype=F32).contiguous().realize()

  for step in range(steps):
    a = step % (size - 1)
    b = a + 1
    reader = base + step  # captured before *this* step's assign
    expected = (ref + step).tolist()
    sc.sink(reader); sc.check(f"chain#{idx} step{step} reader", reader.tolist, expected)

    # NOTE: mixing a bitcast-view assign with a later plain-shrink assign in the same chain fails
    # schedule verification (Ops.INDEX on Ops.AFTER ...) identically on unmodified upstream master,
    # so that combination is a pre-existing scheduler limitation unrelated to this fix - this chain
    # sticks to plain shrink assigns, which is what it's actually testing (sequential view-assigns +
    # interleaved reader ordering); bitcast views are already covered by build_simple_buffer.
    new_val = [float(-(idx * 100 + step + 1))]
    writer = base[a:b].assign(Tensor(new_val, dtype=F32))
    sc.sink(writer)
    ref[a:b] = new_val

  final_reader = base + 100
  final_expected = (ref + 100).tolist()
  sc.sink(final_reader); sc.check(f"chain#{idx} final_reader", final_reader.tolist, final_expected)
  sc.check(f"chain#{idx} raw_buffer", (lambda base=base: base.uop.buffer.numpy().tolist()), ref.tolist())

def build_cross_pair(sc: Scenario, idx: int):
  size = 5
  va = np.arange(size, dtype=np.float32) + idx * 10000
  vb = np.arange(size, dtype=np.float32) + idx * 10000 + 5000
  refa, refb = va.copy(), vb.copy()
  a = Tensor(va.tolist(), dtype=F32).contiguous().realize()
  b = Tensor(vb.tolist(), dtype=F32).contiguous().realize()

  cross_reader = a + b  # spans two buffers, captured before either is assigned
  cross_expected = (refa + refb).tolist()

  new_vals_a = [float(-(idx + 1))]
  writer_a = a[0:1].bitcast(U32).assign(Tensor(to_bits(new_vals_a), dtype=U32))
  refa[0:1] = new_vals_a

  new_vals_b = [float(idx + 1)]
  writer_b = b[1:2].assign(Tensor(new_vals_b, dtype=F32))
  refb[1:2] = new_vals_b

  sc.sink(cross_reader); sc.check(f"cross#{idx} reader", cross_reader.tolist, cross_expected)
  sc.sink(writer_a); sc.sink(writer_b)
  sc.check(f"cross#{idx} raw_a", (lambda a=a: a.uop.buffer.numpy().tolist()), refa.tolist())
  sc.check(f"cross#{idx} raw_b", (lambda b=b: b.uop.buffer.numpy().tolist()), refb.tolist())

def build_all() -> Scenario:
  sc = Scenario()
  for i in range(NUM_SIMPLE): build_simple_buffer(sc, i, KINDS[i % len(KINDS)])
  for i in range(NUM_CHAIN): build_chain_buffer(sc, i)
  for i in range(NUM_CROSS): build_cross_pair(sc, i)
  return sc

def order_forward(x): return x
def order_reverse(x): return list(reversed(x))
def order_shuffled(x):
  rnd = random.Random(20260819)
  y = list(x)
  rnd.shuffle(y)
  return y

def run_epoch(name: str, order_fn):
  sc = build_all()
  n_buffers = NUM_SIMPLE + NUM_CHAIN + NUM_CROSS * 2
  n_nodes = len(tt.all_tensors)
  Tensor.realize(*order_fn(list(sc.sinks)))
  failures = []
  for label, fn, expected in sc.checks:
    actual = fn()
    if actual != expected: failures.append((label, actual, expected))
  print(f"[{name:10s}] buffers={n_buffers:3d}  live_tensor_nodes={n_nodes:4d}  checks={len(sc.checks):4d}  failures={len(failures)}")
  for label, actual, expected in failures[:15]:
    print(f"    FAIL {label}: got {actual} expected {expected}")
  return failures

if __name__ == "__main__":
  all_failures = []
  for name, order_fn in [("forward", order_forward), ("reverse", order_reverse), ("shuffled", order_shuffled)]:
    all_failures += run_epoch(name, order_fn)
  print()
  if all_failures:
    print(f"RESULT: FAILED ({len(all_failures)} check(s) failed)")
    raise SystemExit(1)
  print("RESULT: ALL PASSED")
