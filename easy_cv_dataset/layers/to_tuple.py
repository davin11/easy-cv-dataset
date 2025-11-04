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

from keras.layers import Layer

def boses_to_dense(bounding_boxes, max_boxes, default_value=-1):
    try:
        from tensorflow import RaggedTensor
    except:
        return bounding_boxes
    out_bounding_boxes = dict()
    for key in bounding_boxes.keys():
        if isinstance(bounding_boxes[key], RaggedTensor):
            data_shape = bounding_boxes[key].shape
            out_bounding_boxes[key] = bounding_boxes[key].to_tensor(
                default_value=default_value,
                shape=[data_shape[0], max_boxes] + data_shape[2:],
            )
        else:
            out_bounding_boxes[key] = bounding_boxes[key]
        
    return out_bounding_boxes


class ToTuple(Layer):
    def __init__(self, dictname_input, dictname_target, max_boxes=None, **kwargs):
        super().__init__(**kwargs, trainable=False, autocast=False)
        self.dictname_input = dictname_input
        self.dictname_target = dictname_target
        self.max_boxes = max_boxes

    def call(self, dat):
        x = dat[self.dictname_input]
        y = dat[self.dictname_target]
        if (self.dictname_target == "bounding_boxes") and (self.max_boxes is not None):
            y = boses_to_dense(y, max_boxes=self.max_boxes)
        return x, y


