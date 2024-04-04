# Copyright 2024
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

from sklearn.metrics import precision_recall_curve
from keras_cv import bounding_box


def compute_mAP_metrics(
    model, dataset, bounding_box_format=None, iou_th=0.5, sentinel=-1
):
    from keras.utils import Progbar
    from keras.utils.io_utils import print_msg
    import tensorflow
    import numpy as np

    assert iou_th > 0

    bounding_box_format = (
        bounding_box_format
        if bounding_box_format is not None
        else dataset.bounding_box_format
    )

    if hasattr(dataset, "class_names"):
        class_names = dataset.class_names
    else:
        class_names = None

    list_classes = list()
    list_predictions = list()
    list_scores = list()

    try:
        pbar = Progbar(len(dataset))
        pbar_finalize = False
    except:
        pbar = Progbar(None)
        pbar_finalize = True

    for images, boxes_gt in dataset:
        boxes_pred = model.predict(images, verbose=False)

        for index_image in range(len(images)):
            boxes_pred_image = {
                k: tensorflow.convert_to_tensor(boxes_pred[k][index_image])
                for k in boxes_pred
            }
            target_classes = boxes_gt["classes"][index_image]
            target_valid = target_classes != sentinel
            detection_valid = boxes_pred_image["classes"] != sentinel
            detection_scores = boxes_pred_image["confidence"][detection_valid]
            detection_classes = boxes_pred_image["classes"][detection_valid]

            detection_iou = bounding_box.compute_iou(
                boxes_pred_image["boxes"][detection_valid],
                boxes_gt["boxes"][index_image],
                bounding_box_format=bounding_box_format,
            )

            inds = tensorflow.argsort(detection_scores, direction="DESCENDING")
            for index_box in inds:
                s = detection_scores[index_box]
                c = detection_classes[index_box]

                vald = target_valid & (target_classes == c)
                comb = detection_iou[index_box] * tensorflow.cast(
                    vald, detection_iou.dtype
                )
                th = tensorflow.math.maximum(
                    tensorflow.reduce_max(comb), iou_th
                )
                comb = comb < th
                target_valid = target_valid & comb
                p = tensorflow.reduce_all(comb) == False

                list_classes.append(c.numpy())
                list_scores.append(s.numpy())
                list_predictions.append(p.numpy())

            for c in target_classes[target_valid]:
                list_classes.append(c.numpy())
                list_predictions.append(True)
                list_scores.append(np.nan)

        pbar.add(1)

    if pbar_finalize:
        pbar.update(pbar._seen_so_far, values=None, finalize=True)

    list_classes = np.asarray(list_classes)
    list_predictions = np.asarray(list_predictions)
    list_scores = np.asarray(list_scores)
    min_score = np.nanmin(list_scores)
    list_scores[np.isnan(list_scores)] = min_score - 1.0

    set_classes = sorted(list(set(list_classes)))
    ap = [np.nan] * len(set_classes)
    for index, c in enumerate(set_classes):
        valid = list_classes == c
        if np.any(list_predictions[valid]):
            precision, recall, ths = precision_recall_curve(
                list_predictions[valid], list_scores[valid]
            )
            ths = np.concatenate((ths, [np.PINF]), 0)
            precision[ths < min_score] = 0
            ap[index] = -np.sum(np.diff(recall) * np.array(precision)[:-1])

        name = class_names[int(c)] if class_names else c
        print_msg("%3d) AP of %20s = %7.5f" % (index, name, ap[index]))

    return {"mAP": np.nanmean(ap)}
