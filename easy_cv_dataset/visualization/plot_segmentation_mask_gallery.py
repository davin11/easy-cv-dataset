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

import numpy as np
from keras import ops
from .plot_image_gallery import to_numpy, transform_value_range, _numpy_plot_image_gallery


def _reshape_masks(x):
    shape = tuple(x.shape)
    if (len(shape) == 4) and (shape[-1] != 1):
        x = ops.argmax(x, axis=-1)
        shape = tuple(x.shape)
    if len(shape) == 3:
        x = x[..., None]

    return ops.concatenate((x,x,x), axis=-1)


def plot_segmentation_mask_gallery(
    images,
    num_classes,
    y_true=None,
    y_pred=None,
    value_range=(0,255),
    rows=None,
    cols=None,
    **kwargs
):
    """Plots a gallery of images with corresponding segmentation masks.

    Args:
        images: a Tensor containing images to show in the gallery.
                The images should be batched and of shape (B, H, W, C).
        num_classes: number of segmentation classes.
        y_true: (Optional) a Tensor or NumPy array representing the ground truth
            segmentation masks. The ground truth segmentation maps should be
            batched.
        y_pred: (Optional)  a Tensor or NumPy array representing the predicted
            segmentation masks. The predicted segmentation masks should be
            batched.
        value_range: value range of the images. Default `(0, 255)`.
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

    # Initialize a list to collect the segmentation masks that will be
    # concatenated to the images for visualization.
    masks_to_contatenate = [images, ]

    if y_true is not None:
        plotted_y_true = _reshape_masks(y_true)
        plotted_y_true = transform_value_range(plotted_y_true, (0, num_classes-1), (0, 255))
        plotted_y_true = to_numpy(plotted_y_true).astype("uint8")
        masks_to_contatenate.append(plotted_y_true)
    if y_pred is not None:
        plotted_y_pred = _reshape_masks(y_pred)
        plotted_y_pred = transform_value_range(plotted_y_pred, (0, num_classes-1), (0, 255))
        plotted_y_pred = to_numpy(plotted_y_pred).astype("uint8")
        masks_to_contatenate.append(plotted_y_pred)

    # Concatenate the images and the masks together.
    plotted_images = np.concatenate(masks_to_contatenate, axis=-2)

    return _numpy_plot_image_gallery(images=plotted_images, rows=rows, cols=cols, **kwargs)