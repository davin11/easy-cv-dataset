# Copyright 2023 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
import keras.backend as ops

try:
    import matplotlib.pyplot as plt
except:
    plt = None

def _numpy_plot_image_gallery(
    images,
    scale=2,
    rows=None,
    cols=None,
    show=True,
):
    if plt is None:
        raise ImportError(
            f"Visualization requires the `matplotlib` package. "
            "Please install the package using "
            "`pip install matplotlib`."
        )

    batch_size = len(images)
    rows = rows or int(math.ceil(math.sqrt(batch_size)))
    cols = cols or int(math.ceil(batch_size // rows))

    # Generate subplots
    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(cols * scale, rows * scale),
        frameon=False,
        layout="tight",
        squeeze=True,
        sharex="row",
        sharey="col",
    )
    fig.subplots_adjust(wspace=0, hspace=0)

    if isinstance(axes, np.ndarray) and len(axes.shape) == 1:
        expand_axis = 0 if rows == 1 else -1
        axes = np.expand_dims(axes, expand_axis)
    
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            current_axis = (
                axes[row, col] if isinstance(axes, np.ndarray) else axes
            )
            current_axis.imshow(images[index])
            current_axis.margins(x=0, y=0)
            current_axis.axis("off")

    
    if show:
        plt.show()
    
    return fig 

def to_numpy(x):
    if x is None:
        return None
    x = ops.convert_to_numpy(x)
    # Important for consistency when working with visualization utilities
    return np.ascontiguousarray(x)

def transform_value_range(x, original_range, target_range, dtype=ops.float32):

    if (original_range[0] == target_range[0]) and (original_range[1] == target_range[1]):
        return x

    x = ops.cast(x, dtype=dtype)
    original_min_value = ops.cast(original_range[0], dtype=dtype)
    original_max_value = ops.cast(original_range[1], dtype=dtype)
    target_min_value = ops.cast(target_range[0], dtype=dtype)
    target_max_value = ops.cast(target_range[1], dtype=dtype)

    x = x - original_min_value
    x = x * (target_max_value - target_min_value) / (original_max_value - original_min_value)
    return x + target_min_value


def plot_image_gallery(
    images,
    value_range=(0,255),
    scale=2,
    rows=None,
    cols=None,
    show=True,
):
    """Displays a gallery of images.

    Args:
        images: a Tensor containing images to show in the gallery.
        value_range: value range of the images. Default `(0, 255)`.
        scale: how large to scale the images in the gallery
        rows: (Optional) number of rows in the gallery to show.
            Required if inputs are unbatched.
        cols: (Optional) number of columns in the gallery to show.
            Required if inputs are unbatched.
        show: (Optional) whether to show the gallery of images.

    """
    # Perform image range transform
    images = transform_value_range(
        images, original_range=value_range, target_range=(0, 255)
    )
    images = to_numpy(images).astype("uint8")

    return _numpy_plot_image_gallery(
        images=images, scale=scale,
        rows=rows, cols=cols, show=show)