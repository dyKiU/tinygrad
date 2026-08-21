#!/usr/bin/env python3
"""Model-shaped integration tests for lazy readers followed by view assignments.

Run directly from the tinygrad checkout:

  PYTHONPATH=. python3 extra/view_assign_real_world_models.py

The ring KV-cache and streaming-convolution tests are regressions for the original
prior-reader bug: they fail on base commit 1c3c9e9 and pass on the fix branch.

The packed recurrent-state test describes a known remaining hole. It is marked as
an expected failure until reader discovery follows the underlying storage across
an assignment chain instead of following only the latest AFTER node.
"""

import math
import unittest

import numpy as np

from tinygrad import Tensor


class TestViewAssignRealWorldModels(unittest.TestCase):
  def test_ring_kv_cache_preserves_pending_attention(self):
    """A pending attention result must keep the cache window from before wraparound."""
    keys = np.array([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=np.float32)
    values = np.array([[[[2.0, 1.0], [4.0, 3.0], [8.0, 5.0]]]], dtype=np.float32)
    cache_data = np.stack((keys, values), axis=0)
    cache_kv = Tensor(cache_data).contiguous().realize()

    query_data = np.array([[[[1.0, 0.5]]]], dtype=np.float32)
    query = Tensor(query_data).contiguous().realize()

    # This represents attention for token N. Keep it lazy while token N+1 wraps
    # the ring and reuses slot zero.
    pending_attention = query.scaled_dot_product_attention(cache_kv[0], cache_kv[1])

    replacement_key = np.array([[[[9.0, -4.0]]]], dtype=np.float32)
    replacement_value = np.array([[[[-7.0, 11.0]]]], dtype=np.float32)
    replacement_kv = Tensor(np.stack((replacement_key, replacement_value), axis=0))
    cache_kv[:, :, :, 0:1, :].assign(replacement_kv).realize()

    scores = query_data @ keys.swapaxes(-1, -2) / math.sqrt(query_data.shape[-1])
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected_attention = (weights / weights.sum(axis=-1, keepdims=True)) @ values
    np.testing.assert_allclose(pending_attention.numpy(), expected_attention, rtol=1e-6, atol=1e-6)

    expected_cache = cache_data.copy()
    expected_cache[:, :, :, 0:1, :] = np.stack((replacement_key, replacement_value), axis=0)
    np.testing.assert_allclose(cache_kv.numpy(), expected_cache, rtol=0, atol=0)

  def test_streaming_causal_conv_preserves_pending_output(self):
    """A lazy causal-convolution output must survive a history shift and append."""
    history_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    history = Tensor(history_data).contiguous().realize()
    kernel_data = np.array([1.0, 10.0, 100.0, 1000.0], dtype=np.float32)
    kernel = Tensor(kernel_data.reshape(1, 1, 1, -1)).contiguous().realize()

    # One valid Conv1D output, expressed as a height-one Conv2D. It remains lazy
    # while the persistent history buffer advances to the next streaming step.
    pending_output = history.reshape(1, 1, 1, -1).conv2d(kernel).reshape(())

    history[:-1].assign(history[1:] * 1)
    history[-1:].assign(Tensor([5.0]))
    history.realize()

    expected_output = float((history_data * kernel_data).sum())
    self.assertAlmostEqual(pending_output.item(), expected_output, places=5)
    np.testing.assert_allclose(history.numpy(), [2.0, 3.0, 4.0, 5.0], rtol=0, atol=0)

  @unittest.expectedFailure
  def test_packed_recurrent_state_disjoint_then_overlapping_update(self):
    """An old recurrent-state reader must survive a disjoint write followed by an overlap."""
    # Layout: [two convolution-history values | one 2x2 recurrent matrix].
    packed_data = np.array([11.0, 12.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    packed_state = Tensor(packed_data).contiguous().realize()
    recurrent_state = packed_state[2:].reshape(2, 2)
    recurrent_input_data = np.array([2.0, 3.0], dtype=np.float32)
    recurrent_input = Tensor(recurrent_input_data).contiguous().realize()

    # Keep an output from the old recurrent matrix lazy. Updating the convolution
    # field first is disjoint, but the following recurrent update overlaps it.
    pending_output = recurrent_state @ recurrent_input
    packed_state[:2].assign(Tensor([21.0, 22.0]))
    replacement_recurrent = np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32)
    packed_state[2:].assign(Tensor(replacement_recurrent.reshape(-1)))
    packed_state.realize()

    expected_output = packed_data[2:].reshape(2, 2) @ recurrent_input_data
    np.testing.assert_allclose(pending_output.numpy(), expected_output, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(packed_state.numpy(), [21.0, 22.0, 4.0, 5.0, 6.0, 7.0], rtol=0, atol=0)


if __name__ == "__main__":
  unittest.main(verbosity=2)
