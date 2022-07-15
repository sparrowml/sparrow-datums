# Sparrow Datums

Sparrow Datums is a Python package for vision AI data structures, related operations and serialization/deserialization.
Specifically, it makes it easier to work with bounding boxes, key points (TODO), and segmentation masks (TODO).
It supports individual objects, frames of objects, multiple frames of objects, objects augmented with class labels and confidence scores, and more.

Sparrow Datums also supports object tracking where the identity of the object is maintained. And that data
can be streamed instead of keeping it all in a single file.

# Quick Start Example

## Installation

```bash
pip install -U sparrow-datums
```

## Switching between box parameterizations

```python
import numpy as np
from sparrow_datums import FrameBoxes, PType

boxes = FrameBoxes(np.ones((4, 4)), PType.absolute_tlwh)
boxes.to_tlbr()

# Expected result
# FrameBoxes([[1., 1., 2., 2.],
#             [1., 1., 2., 2.],
#             [1., 1., 2., 2.],
#             [1., 1., 2., 2.]])
```

## Slicing

Notice that all "chunk" objects override basic NumPy arrays. This means that some filtering operations work as expected:

```python
boxes[:2]

# Expected result
# FrameBoxes([[1., 1., 1., 1.],
#             [1., 1., 1., 1.]])
```

But sub-types do their own validation. For example, `FrameBoxes` must be a `(n, 4)` array. Therefore, selecting a single column throws an error:

```python
boxes[:, 0]

# Expected exception
# ValueError: A frame boxes object must be a 2D array
```

Instead, chunks expose different subsets of the data as properties. For example, you can get the `x` coordinate as an array:

```python
boxes.x

# Expected result
# array([1., 1., 1., 1.])
```

Or the width of the boxes:

```python
boxes.w

# Expected result
# array([1., 1., 1., 1.])
```

If you need to access the raw data, you can do that with a chunk's `array` property:

```python
boxes.array[0, 0]

# Expected result
# 1.0
```

## Operations

Sparrow Datums comes with common operations for data types. For example, you can compute the pairwise IoU of two sets of `FrameBoxes`:

```python
from sparrow_datums import pairwise_iou

pairwise_iou(boxes, boxes + 0.1)

# array([[0.57857143, 0.57857143, 0.57857143, 0.57857143],
#        [0.57857143, 0.57857143, 0.57857143, 0.57857143],
#        [0.57857143, 0.57857143, 0.57857143, 0.57857143],
#        [0.57857143, 0.57857143, 0.57857143, 0.57857143]])
```