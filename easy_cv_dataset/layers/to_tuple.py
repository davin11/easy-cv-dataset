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


import keras
from keras_cv import bounding_box
BOUNDING_BOXES = "bounding_boxes"

def _dict_to_tuple_fun(dat, dictname_input, dictname_target, max_boxes=None):
    x = dat[dictname_input]
    y = dat[dictname_target]
    if dictname_target == BOUNDING_BOXES:
        y = bounding_box.to_dense(y, max_boxes=max_boxes)
    return x, y

class ToTuple(keras.layers.Layer):
    def __init__(self, dictname_input, dictname_target, max_boxes=None, **kwargs):
        super().__init__(**kwargs, trainable=False, autocast=False)
        self.dictname_input = dictname_input
        self.dictname_target = dictname_target
        self.max_boxes = max_boxes

    def call(self, dat):
        x = dat[self.dictname_input]
        y = dat[self.dictname_target]
        if self.dictname_target == BOUNDING_BOXES:
            y = bounding_box.to_dense(y, max_boxes=self.max_boxes)
        return x, y