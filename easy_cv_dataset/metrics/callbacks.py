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

from keras.callbacks import Callback

class EvaluateMAPmetricsCallback(Callback):
    def __init__(self, data, bounding_box_format=None, iou_th=0.5):
        super().__init__()
        self.data = data
        self.bounding_box_format = bounding_box_format
        self.iou_th = iou_th

    def on_epoch_end(self, epoch, logs):
        from keras_cv.metrics.object_detection.compute_map_metrics import (
            compute_mAP_metrics,
        )

        metrics = compute_mAP_metrics(
            self.model,
            self.data,
            bounding_box_format=self.bounding_box_format,
            iou_th=self.iou_th,
        )
        logs.update(metrics)
        return logs
