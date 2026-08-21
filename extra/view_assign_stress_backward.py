#!/usr/bin/env python3
"""Stress: view-assign x autograd interactions (companion to view_assign_stress_graph.py).

Run directly (not pytest):  PYTHONPATH=. python3 extra/view_assign_stress_backward.py

Two sections:
  MUST PASS    - scenarios the bugfix branch is supposed to handle; any failure here exits 1.
  KNOWN HOLES  - regressions found while reviewing the fix (2026-08-21). These currently FAIL
                 on the bugfix branch but WORKED (or at least didn't crash) on upstream base.
                 The script prints their live status; they don't affect the exit code, so this
                 script stays useful both before and after the holes get fixed.

The two known holes, in plain words:
  HOLE 1: after `w[:1].assign(...)` with ANY tensor previously reading w (even an int cast),
          calling backward() on a NEW loss built after the assign crashes with
          "failed to compute gradient for Ops.AFTER" - unless w was realized in between.
          Cause: backward() registers w's PRE-assign uop in _snapshot_grad_owners; that same
          uop sits inside every post-assign AFTER graph, so backward() adds it as an extra
          gradient target and the gradient engine can't differentiate through AFTER/STORE.
          Realistic trigger: gradient-accumulation loops that assign without realizing.
  HOLE 2: loss.gradient(w) (the public API, as opposed to loss.backward()) on a prior loss
          silently returns ZEROS after the assign snapshots the loss, because gradient() knows
          nothing about the _snapshot_grad_owners remapping that backward() uses.
"""
import gc
from tinygrad import Tensor, dtypes

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
  """fn asserts the CORRECT behavior. `currently` describes today's broken behavior."""
  try:
    fn()
    hole_status.append((name, "FIXED"))
    print(f"  HOLE FIXED  {name}")
  except Exception as e:
    hole_status.append((name, f"open ({type(e).__name__})"))
    print(f"  hole open   {name} -> {type(e).__name__}: {str(e).splitlines()[0][:90]}")
    print(f"              (currently: {currently})")

# ---------------- MUST PASS ----------------
print("MUST PASS")

def prior_loss_backward():
  w = Tensor([1.0, 2.0]).contiguous().realize()
  loss = (w * w).sum()
  w[:1].assign(Tensor([5.0]))
  loss.backward()
  assert loss.item() == 5.0
  assert w.grad.tolist() == [2.0, 4.0], w.grad.tolist()
check("prior loss backward sees pre-assign values", prior_loss_backward)

def prior_loss_backward_twice():
  w = Tensor([1.0, 2.0]).contiguous().realize()
  loss = (w * w).sum()
  w[:1].assign(Tensor([5.0]))
  loss.backward(); loss.backward()
  assert w.grad.tolist() == [4.0, 8.0], w.grad.tolist()
check("prior loss backward twice accumulates", prior_loss_backward_twice)

def post_loss_after_realize():
  w = Tensor([1.0, 2.0]).contiguous().realize()
  _ = (w * 3).sum()                       # prior reader triggers the snapshot machinery
  w[:1].assign(Tensor([5.0]))
  w.realize()                             # workaround for HOLE 1
  loss = (w * w).sum()
  loss.backward()
  assert w.grad.tolist() == [10.0, 4.0], w.grad.tolist()
check("post-assign loss backward (after realize)", post_loss_after_realize)

def alias_grads():
  w = Tensor([1.0, 2.0, 3.0, 4.0]).contiguous().realize()
  v = w.reshape(2, 2)
  loss = (v * v).sum()
  w[:1].assign(Tensor([9.0]))
  loss.backward()
  assert w.grad.tolist() == [2.0, 4.0, 6.0, 8.0], w.grad.tolist()
  assert v.grad.tolist() == [[2.0, 4.0], [6.0, 8.0]], v.grad.tolist()
check("reshape alias gets its own pre-assign grads", alias_grads)

def disjoint_reader_loss():
  w = Tensor([1.0, 2.0, 3.0, 4.0]).contiguous().realize()
  loss = (w[:2] * w[:2]).sum()
  w[2:].assign(Tensor([8.0, 9.0]))
  loss.backward()
  assert w.grad.tolist() == [2.0, 4.0, 0.0, 0.0], w.grad.tolist()
check("disjoint prior loss backward", disjoint_reader_loss)

def interleaved_losses():
  w = Tensor([1.0, 2.0]).contiguous().realize()
  l1 = (w * w).sum()
  w[:1].assign(Tensor([5.0]))
  l2 = (w * w).sum()
  l1.backward()
  assert w.grad.tolist() == [2.0, 4.0], w.grad.tolist()
check("first of two interleaved losses", interleaved_losses)

def owner_replaced():
  w = Tensor([1.0, 2.0]).contiguous().realize()
  loss = (w * w).sum()
  w[:1].assign(Tensor([5.0]))
  w.replace(Tensor([100.0, 200.0]).realize())
  loss.backward()
  assert w.grad is None, "stale grad routed to replaced tensor"
check("replaced owner gets no stale grad", owner_replaced)

def owner_dead():
  w = Tensor([1.0, 2.0]).contiguous().realize()
  loss = (w * w).sum()
  w[:1].assign(Tensor([5.0]))
  del w
  gc.collect()
  loss.backward()
check("backward with dead owner doesn't crash", owner_dead)

def training_loop_with_realize():
  w = Tensor([1.0, 2.0]).contiguous().realize()
  for _ in range(3):
    loss = (w * w).sum()
    w.grad = None
    loss.backward()
    w[:].assign(w - 0.1 * w.grad)
    w.realize()
  expect = [1.0, 2.0]
  for _ in range(3): expect = [x - 0.2 * x for x in expect]
  assert all(abs(a - b) < 1e-5 for a, b in zip(w.tolist(), expect)), (w.tolist(), expect)
check("manual sgd loop via full-view assign (with realize)", training_loop_with_realize)

def explicit_gradient_arg():
  w = Tensor([1.0, 2.0]).contiguous().realize()
  out = w * w
  w[:1].assign(Tensor([5.0]))
  out.backward(gradient=Tensor([1.0, 1.0]))
  assert w.grad.tolist() == [2.0, 4.0], w.grad.tolist()
check("backward with explicit gradient tensor", explicit_gradient_arg)

# ---------------- KNOWN HOLES ----------------
print("KNOWN HOLES (found in review; upstream base did NOT crash on these)")

def hole1_post_loss_no_realize():
  w = Tensor([1.0, 2.0]).contiguous().realize()
  _ = w + 1.0                              # any prior reader, float or not
  w[:1].assign(Tensor([5.0]))
  loss = (w * w).sum()                     # built after the assign, no realize in between
  loss.backward()                          # <- crashes: failed to compute gradient for Ops.AFTER
  assert w.grad.tolist() == [10.0, 4.0], w.grad.tolist()
hole("HOLE 1: post-assign loss backward without realize", hole1_post_loss_no_realize,
     "raises RuntimeError; workaround: w.realize() between assign and building the new loss")

def hole1b_accumulation_loop():
  w = Tensor([1.0, 2.0]).contiguous().realize()
  for step in range(3):
    loss = (w * w).sum()
    loss.backward()                        # crashes at step 1 (second iteration)
    w[:1].assign(Tensor([float(step + 5)]))
  assert w.grad is not None
hole("HOLE 1b: grad-accumulation loop, assign without realize", hole1b_accumulation_loop,
     "raises RuntimeError on the second iteration")

def hole2_gradient_api():
  w = Tensor([1.0, 2.0]).contiguous().realize()
  loss = (w * w).sum()
  w[:1].assign(Tensor([5.0]))
  g = loss.gradient(w)[0].tolist()
  # backward() routes this correctly to [2.0, 4.0]; gradient() should agree, not return zeros
  assert g == [2.0, 4.0], f"gradient() returned {g}"
hole("HOLE 2: loss.gradient(w) on snapshotted prior loss", hole2_gradient_api,
     "silently returns [0.0, 0.0]")

print()
for name, st in hole_status: print(f"  {st:24s} {name}")
if must_pass_failures:
  print(f"RESULT: FAILED ({len(must_pass_failures)} must-pass failure(s))")
  raise SystemExit(1)
print("RESULT: ALL MUST-PASS OK")
