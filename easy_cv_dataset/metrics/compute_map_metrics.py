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
import tensorflow as tf
import numpy as np
from keras.utils import Progbar
from keras.callbacks import Callback

def compute_mAP_metrics(
    model, dataset, bounding_box_format=None, iou_th=0.5, sentinel=-1
):

    assert iou_th > 0

    if bounding_box_format is None:
        bounding_box_format = dataset.bounding_box_format

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
                k: tf.convert_to_tensor(boxes_pred[k][index_image])
                for k in boxes_pred
            }
            target_classes = boxes_gt["classes"][index_image]
            target_valid = target_classes != sentinel
            detection_valid = boxes_pred_image["classes"] != sentinel
            detection_scores = boxes_pred_image["confidence"][detection_valid]
            detection_classes = tf.cast(boxes_pred_image["classes"][detection_valid],
                                        target_classes.dtype)

            detection_iou = bounding_box.compute_iou(
                boxes_pred_image["boxes"][detection_valid],
                boxes_gt["boxes"][index_image],
                bounding_box_format=bounding_box_format,
            )

            inds = tf.argsort(detection_scores, direction="DESCENDING")
            for index_box in inds:
                s = detection_scores[index_box]
                c = detection_classes[index_box]

                vald = target_valid & (target_classes == c)
                comb = detection_iou[index_box] * tf.cast(
                    vald, detection_iou.dtype
                )
                th = tf.math.maximum(
                    tf.reduce_max(comb), iou_th
                )
                comb = comb < th
                target_valid = target_valid & comb
                p = tf.reduce_all(comb) == False

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
    if np.all(np.isnan(list_scores)):
        min_score = 0.0
    else:
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
            ths = np.concatenate((ths, [np.inf]), 0)
            precision[ths < min_score] = 0
            ap[index] = -np.sum(np.diff(recall) * np.array(precision)[:-1])
            
        try:
            name = class_names[int(c)] if class_names else c
        except:
            name = c
        
        print("%3d) AP of %20s = %7.5f" % (index, name, ap[index]))

    return np.nanmean(ap)

class EvaluateMAPmetricsCallback(Callback):
    def __init__(self, data, bounding_box_format=None, iou_th=0.5):
        super().__init__()
        self.data = data
        self.bounding_box_format = bounding_box_format
        self.iou_th = iou_th

    def on_epoch_end(self, epoch, logs):
        metrics = compute_mAP_metrics(
            self.model,
            self.data,
            bounding_box_format=self.bounding_box_format,
            iou_th=self.iou_th,
        )
        logs.update({'mAP': metrics})
        return logs


def compute_dataset_pycoco_metrics(model, dataset, bounding_box_format):
    from keras_cv.callbacks import PyCOCOCallback
    fun = PyCOCOCallback(validation_data=dataset, bounding_box_format=bounding_box_format)
    fun.set_model(model)
    logs = fun.on_epoch_end(0)
    return logs
