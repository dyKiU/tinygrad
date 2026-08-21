#!/usr/bin/env python3
"""Stress: view-assign under TinyJit (companion to view_assign_stress_graph.py).

Run directly (not pytest):  PYTHONPATH=. python3 extra/view_assign_stress_jit.py

Exercises the _capture_effects machinery the bugfix added: assigns inside a jitted function
must be captured and replayed even when nothing realizes them explicitly, mid-capture realizes
must not reorder writes vs reads, and an exception during capture must not leave stale effects
poisoning the next jit. Every call is verified for all of: warmup (call 0), capture (call 1)
and replay (calls 2+).
"""
from tinygrad import Tensor, TinyJit, nn
from tinygrad.helpers import Context

failures = []
def check(name, fn):
  try:
    fn()
    print(f"  pass  {name}")
  except Exception as e:
    failures.append(name)
    print(f"  FAIL  {name}: {type(e).__name__}: {str(e).splitlines()[0][:110]}")

def jit_train_step():
  w = Tensor([[1.0, 2.0], [3.0, 4.0]]).contiguous().realize()
  opt = nn.optim.SGD([w], lr=0.1)
  @TinyJit
  def step(x):
    opt.zero_grad()
    loss = (x @ w).sum()
    loss.backward()
    opt.step()
    return loss.realize()
  losses = []
  with Context(TRAINING=1):
    for _ in range(5): losses.append(step(Tensor([[1.0, 1.0]])).item())
  assert losses == sorted(losses, reverse=True) and losses[0] > losses[-1], losses
check("jitted optimizer training step unaffected", jit_train_step)

def jit_kv_cache():
  cache = Tensor.zeros(4, 2).contiguous().realize()
  @TinyJit
  def upd(pos_val: Tensor):
    cache[1:2].assign(pos_val)
    return (cache * 1).sum().realize()
  outs = [upd(Tensor([[float(i + 1)] * 2])).item() for i in range(4)]
  assert outs == [2.0, 4.0, 6.0, 8.0], outs
  assert cache.tolist() == [[0.0, 0.0], [4.0, 4.0], [0.0, 0.0], [0.0, 0.0]], cache.tolist()
check("kv-cache-style persistent view assign in jit", jit_kv_cache)

def jit_effect_only():
  @TinyJit
  def f(x: Tensor):
    x[:1].assign(Tensor([9.0], device=x.device))
  for i in range(4):
    x = Tensor([float(i + 1), float(i + 2)]).contiguous().realize()
    assert f(x) is None
    assert x.tolist() == [9.0, float(i + 2)], (i, x.tolist())
check("effect-only jit (nothing returned) still writes", jit_effect_only)

def jit_two_assigns_effect_only():
  @TinyJit
  def f(x: Tensor, a: Tensor, b: Tensor):
    x[:1].assign(a)
    x[1:].assign(b)
  for i in range(4):
    x = Tensor([0.0, 0.0]).contiguous().realize()
    f(x, Tensor([float(i)]).realize(), Tensor([float(i * 10)]).realize())
    assert x.tolist() == [float(i), float(i * 10)], (i, x.tolist())
check("two view assigns to one buffer, effect-only jit", jit_two_assigns_effect_only)

def jit_mid_capture_realize():
  @TinyJit
  def f(x: Tensor, y: Tensor):
    x[:1].assign(Tensor([9.0], device=x.device))
    z = (y * 2).realize()          # unrelated realize while the assign effect is pending
    return (x + z).realize()       # must read post-assign x
  for i in range(4):
    x = Tensor([float(i + 1), float(i + 2)]).contiguous().realize()
    out = f(x, Tensor([10.0, 20.0]).contiguous().realize()).tolist()
    assert out == [29.0, float(i + 2) + 40.0], (i, out)
    assert x.tolist() == [9.0, float(i + 2)], (i, x.tolist())
check("mid-capture unrelated realize keeps write ordering", jit_mid_capture_realize)

def jit_prior_and_post_readers():
  @TinyJit
  def f(x: Tensor):
    prior = x + 100
    x[:1].assign(Tensor([7.0], device=x.device))
    post = x + 0.5
    return prior.realize(), post.realize()
  for i in range(4):
    x = Tensor([float(i + 1), float(i + 2)]).contiguous().realize()
    prior, post = f(x)
    assert prior.tolist() == [float(i + 101), float(i + 102)], (i, prior.tolist())
    assert post.tolist() == [7.5, float(i + 2) + 0.5], (i, post.tolist())
check("prior reader isolated + post reader sees write, per replay", jit_prior_and_post_readers)

def jit_rhs_is_prior_reader():
  @TinyJit
  def f(x: Tensor):
    reader = x[:1] + 10
    x[:1].assign(reader)
    return reader
  for i in range(4):
    x = Tensor([float(i + 1), float(i + 2)]).contiguous().realize()
    assert f(x).tolist() == [float(i + 11)], i
    assert x.tolist() == [float(i + 11), float(i + 2)], (i, x.tolist())
check("assign rhs reads the assigned storage (rmw) in jit", jit_rhs_is_prior_reader)

def jit_exception_cleanup():
  @TinyJit
  def bad(x: Tensor):
    x[:1].assign(Tensor([1.0], device=x.device))
    raise ValueError("boom")
  for _ in range(3):
    try: bad(Tensor([1.0, 2.0]).contiguous().realize())
    except Exception as e:
      assert "boom" in str(e), e
  import tinygrad.tensor as tt
  assert len(tt._capture_effects) == 0, "stale capture effects leaked"
  @TinyJit
  def good(y: Tensor): return (y + 1).realize()
  for i in range(3):
    assert good(Tensor([float(i)]).realize()).tolist() == [float(i + 1)]
check("exception during capture leaves no stale effects", jit_exception_cleanup)

print()
if failures:
  print(f"RESULT: FAILED ({len(failures)} failure(s))")
  raise SystemExit(1)
print("RESULT: ALL PASSED")
