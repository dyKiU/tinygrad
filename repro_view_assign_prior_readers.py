#!/usr/bin/env python3
# A lazy reader built before a view assign reads the post-write value when co-realized.
# Expected: reader [11, 12] (values at construction time), x [9, 2]. Observed: reader [19, 12].
# Same result in both co-realization orders, on CPU and METAL.
import sys
from tinygrad import Tensor, dtypes

ok = True
for order in ("reader_first", "writer_first"):
  x = Tensor([1, 2], dtype=dtypes.int32).contiguous().realize()
  reader = x + 10
  writer = x[:1].assign(Tensor([9], dtype=dtypes.int32))
  Tensor.realize(*((reader, writer) if order == "reader_first" else (writer, reader)))
  print(f"{order}: reader={reader.tolist()} expected [11, 12] | x={x.tolist()} expected [9, 2]")
  ok = ok and reader.tolist() == [11, 12] and x.tolist() == [9, 2]

sys.exit(0 if ok else 1)
