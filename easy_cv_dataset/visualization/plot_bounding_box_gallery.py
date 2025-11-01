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

import functools
import numpy as np
import keras.backend as ops
from .plot_image_gallery import to_numpy, transform_value_range, _numpy_plot_image_gallery
from keras.utils.bounding_boxes import convert_format as convert_boxes_format

try:
    import cv2
except:
    cv2 = None

def _draw_bounding_boxes(
    images,
    bounding_boxes,
    color,
    line_thickness=1,
    font_scale=1.0,
    text_thickness=None,
    class_mapping=None,
):
    if cv2 is None:
        raise ImportError(
            f"plot_bounding_box_gallery requires the `cv2` package. "
            "Please install the package using "
            "`pip install opencv-python`."
        )
    
    text_thickness = text_thickness or line_thickness
    outline_factor = images[0].shape[-2] // 100
    class_mapping = class_mapping or {}
    result = list()

    for i in range(len(images)):
        image = images[i]
        boxes = bounding_boxes["boxes"][i]
        classes = bounding_boxes["classes"][i]
        if "confidence" in bounding_boxes:
            confidence = bounding_boxes["confidence"][i]
        else:
            confidence = None

        
        for b_id in range(len(bounding_box_batch["boxes"])):
            x1, y1, x2, y2 = boxes[b_id]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(classes[b_id])
            confid = confidence[b_id] if confidence else None

            if class_id < 0:
                continue
            
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 0, 0, 0.5),
                line_thickness + outline_factor,
            )
            cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

            label = class_mapping[class_id] if class_id in class_mapping else '%d' % class_id
            if confid is not None:
                label = f"{label} | {confid:.2f}"

            x, y = _find_text_location(
                x, y, font_scale, line_thickness, outline_factor
            )
            cv2.putText(
                image,
                label,
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0, 0.5),
                text_thickness + outline_factor,
            )
            cv2.putText(
                image,
                label,
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                text_thickness,
            )
        result.append(image)
    return result


def _find_text_location(x, y, font_scale, line_thickness, outline_factor):
    font_height = int(font_scale * 12)
    target_y = y - int(8 + outline_factor)
    if target_y - (2 * font_height) > 0:
        return x, y - int(8 + outline_factor)

    line_offset = line_thickness + outline_factor
    static_offset = 3

    return (
        x + outline_factor + static_offset,
        y + (2 * font_height) + line_offset + static_offset,
    )

def plot_bounding_box_gallery(
    images,
    bounding_box_format='xyxy',
    y_true=None,
    y_pred=None,
    value_range=(0,255),
    class_mapping=None,
    true_color=(0, 188, 212),
    pred_color=(255, 235, 59),
    line_thickness=2,
    font_scale=1.0,
    text_thickness=None,
    rows=None,
    cols=None,
    **kwargs
):
    """Plots a gallery of images with corresponding bounding box annotations.

    Args:
        images: a Tensor containing images to show in the gallery.
        bounding_box_format: the bounding_box_format the provided bounding boxes
            are in.
        y_true: (Optional) a Keras bounding box dictionary representing the
            ground truth bounding boxes.
        y_pred: (Optional) a Keras bounding box dictionary representing the
            predicted bounding boxes.
        value_range: value range of the images. Default `(0, 255)`.
        pred_color: three element tuple representing the color to use for
            plotting predicted bounding boxes.
        true_color: three element tuple representing the color to use for
            plotting true bounding boxes.
        class_mapping: (Optional) class mapping from class IDs to strings
        line_thickness: (Optional) line_thickness for the box and text labels.
            Defaults to 2.
        text_thickness: (Optional) the line_thickness for the text, defaults to
            `1.0`.
        font_scale: (Optional) font size to draw bounding boxes in.
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
    plotted_images = to_numpy(images).astype("uint8")

    if y_true is not None:
        y_true = y_true.copy()
        y_true = convert_boxes_format(
            y_true, source=bounding_box_format, target="xyxy"
        )
        y_true["boxes"] = to_numpy(y_true["boxes"])
        y_true["classes"] = to_numpy(y_true["classes"])
        plotted_images = _draw_bounding_boxes(
            plotted_images,
            y_true,
            true_color,
            line_thickness=line_thickness,
            text_thickness=text_thickness,
            font_scale=font_scale,
            class_mapping=class_mapping,
        )

    if y_pred is not None:
        y_pred = y_pred.copy()
        y_pred = convert_boxes_format(
            y_pred, source=bounding_box_format, target="xyxy"
        )
        y_pred["boxes"] = to_numpy(y_pred["boxes"])
        y_pred["classes"] = to_numpy(y_pred["classes"])
        plotted_images = _draw_bounding_boxes(
            plotted_images,
            y_pred,
            pred_color,
            line_thickness=line_thickness,
            text_thickness=text_thickness,
            font_scale=font_scale,
            class_mapping=class_mapping,
        )

    return _numpy_plot_image_gallery(images=plotted_images, rows=rows, cols=cols, **kwargs)