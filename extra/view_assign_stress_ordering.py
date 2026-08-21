#!/usr/bin/env python3
"""Stress: multi-assign ordering vs prior readers (companion to view_assign_stress_graph.py).

Run directly (not pytest):  PYTHONPATH=. python3 extra/view_assign_stress_ordering.py

Targets one specific weakness found while reviewing the fix (2026-08-21):

  The per-assign reader scan anchors on the CURRENT chain head (the AFTER node from the
  previous assign), not on the underlying storage. A prior reader that assign #1 classified
  as disjoint keeps pointing at the raw buffer; assign #2's scan can't see it (its graph
  doesn't contain the new chain head), so a second write that DOES overlap the reader
  corrupts it silently. This breaks the fix's own core guarantee ("a prior reader keeps the
  value it captured, in any realize order"). Upstream base was also wrong here (it protected
  nothing), so it's a hole rather than a regression - but it's exactly the class of bug the
  fix exists to prevent, and it produces silently wrong VALUES, not an error.

MUST PASS cases document what the fix does handle; KNOWN HOLES track the miss.
"""
from tinygrad import Tensor

must_pass_failures = []
hole_status = []

def check(name, fn):
  try:
    fn()
    print(f"  pass  {name}")
  except Exception as e:
    must_pass_failures.append(name)
    print(f"  FAIL  {name}: {type(e).__name__}: {str(e).splitlines()[0][:110]}")

def hole(name, fn, currently):
  try:
    fn()
    hole_status.append((name, "FIXED"))
    print(f"  HOLE FIXED  {name}")
  except Exception as e:
    hole_status.append((name, f"open ({type(e).__name__})"))
    print(f"  hole open   {name}")
    print(f"              (currently: {currently})")

print("MUST PASS")

def overlapped_then_overlapped():
  x = Tensor([1.0, 2.0, 3.0, 4.0]).contiguous().realize()
  r = x[:2] * 1
  x[1:2].assign(Tensor([9.0]))        # overlaps r -> snapshot protects it
  x[:1].assign(Tensor([7.0]))         # overlaps again - snapshot still holds
  x.realize()
  assert x.tolist() == [7.0, 9.0, 3.0, 4.0], x.tolist()
  assert r.tolist() == [1.0, 2.0], r.tolist()
check("reader overlapping write1 stays protected through write2", overlapped_then_overlapped)

def mid_reader_overlapped():
  x = Tensor([1.0, 2.0, 3.0, 4.0]).contiguous().realize()
  x[3:].assign(Tensor([9.0]))
  r = x[:1] * 1                       # created after write1: anchors on the chain head
  x[:1].assign(Tensor([7.0]))         # overlaps r -> detected, snapshotted
  x.realize()
  assert r.tolist() == [1.0], r.tolist()
check("reader created between writes is protected", mid_reader_overlapped)

def disjoint_stays_lazy_and_correct():
  x = Tensor([1.0, 2.0, 3.0, 4.0]).contiguous().realize()
  r = x[:1] * 1
  x[3:].assign(Tensor([9.0]))         # disjoint: r intentionally left unsnapshotted
  x.realize()
  assert r.tolist() == [1.0], r.tolist()
check("single disjoint write leaves reader correct", disjoint_stays_lazy_and_correct)

print("KNOWN HOLES")

def hole3_disjoint_then_overlapped():
  x = Tensor([1.0, 2.0, 3.0, 4.0]).contiguous().realize()
  r = x[:1] * 1                       # reads element 0
  x[3:].assign(Tensor([9.0]))         # write1 disjoint from r -> r keeps raw buffer ref
  x[:1].assign(Tensor([7.0]))         # write2 overlaps r but the scan can't see r anymore
  x.realize()
  assert x.tolist() == [7.0, 2.0, 3.0, 9.0], x.tolist()
  assert r.tolist() == [1.0], f"STALE READ: {r.tolist()}"
hole("HOLE 3: disjoint-from-write1 reader corrupted by overlapping write2",
     hole3_disjoint_then_overlapped, "r silently reads 7.0 instead of 1.0")

def hole3b_deep_chain():
  x = Tensor([1.0, 2.0, 3.0, 4.0]).contiguous().realize()
  r = x[:1] * 1
  x[3:].assign(Tensor([9.0]))
  x[2:3].assign(Tensor([8.0]))
  x[:1].assign(Tensor([7.0]))
  x.realize()
  assert r.tolist() == [1.0], f"STALE READ: {r.tolist()}"
hole("HOLE 3b: same miss through a 3-assign chain", hole3b_deep_chain,
     "r silently reads 7.0 instead of 1.0")

print()
for name, st in hole_status: print(f"  {st:24s} {name}")
if must_pass_failures:
  print(f"RESULT: FAILED ({len(must_pass_failures)} must-pass failure(s))")
  raise SystemExit(1)
print("RESULT: ALL MUST-PASS OK")
