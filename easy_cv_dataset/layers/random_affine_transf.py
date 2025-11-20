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

import numpy as np
import tensorflow as tf
import keras
from keras.src.layers.preprocessing.image_preprocessing import BaseImagePreprocessingLayer 
from keras.utils.bounding_boxes import convert_format as convert_boxes_format
from keras.utils.bounding_boxes import affine_transform as affine_boxes_transform
from keras.utils.bounding_boxes import clip_to_image_size
from keras.random import SeedGenerator


def get_range(x, name):
    if isinstance(x, (float, int)):
        if x==0:
            return None
        y = [- x, + x]
    elif len(x) == 2 and all(
        isinstance(val, (float, int)) for val in x
    ):
        y = [x[0], x[1]]
    else:
        raise ValueError(
            f"`{name}` should be a float or "
            "a tuple or list of two floats. "
            f"Received: {x}"
        )
    return y


class RandomAffineTransf(BaseImagePreprocessingLayer):
    """A preprocessing layer which randomly applies an affine trasformation during training.

    This layer will apply random affine trasformation to each image, filling empty space
    according to `fill_mode`.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of interger or floating point dtype. By default, the layer will output
    floats.

    Input shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format

    Output shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format

    Arguments:
        rotation_range: Int. Degree range for random rotations.
        zoom_range: Float or [lower, upper]. Range for random zoom. When
          represented as a single float, this value is used for both the upper and
          lower bound. A positive value means zooming out, while a negative value
          means zooming in. For instance, `zoom_range=(-0.1, 0.3)` result in an
          output zoomed out by a random amount in the range `[-10%, +30%]`.
        width_shift_range: Float fraction of total width
        height_shift_range: Float fraction of total height
        shear_range: Float. Shear Intensity (Shear angle in counter-clockwise
          direction in degrees)
        horizontal_flip: Boolean. Randomly flip inputs horizontally.
        vertical_flip: Boolean. Randomly flip inputs vertically.
        fill_mode: Points outside the boundaries of the input are filled according
          to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
          - *reflect*: `(d c b a | a b c d | d c b a)` The input is extended by
            reflecting about the edge of the last pixel.
          - *constant*: `(k k k k | a b c d | k k k k)` The input is extended by
            filling all values beyond the edge with the same constant value k = 0.
          - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
            wrapping around to the opposite edge.
          - *nearest*: `(a a a a | a b c d | d d d d)` The input is extended by
            the nearest pixel.
        interpolation: Interpolation mode. Supported values: `"nearest"`,
          `"bilinear"`.
        seed: Integer. Used to create a random seed.
        fill_value: a float represents the value to be filled outside the
          boundaries when `fill_mode="constant"`.
        bounding_box_format: The format of bounding boxes of input dataset. Refer
          https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
          for more details on supported bounding box formats.
          
    """

    _SUPPORTED_FILL_MODE = ("reflect", "wrap", "constant", "nearest")
    _SUPPORTED_INTERPOLATION = ("nearest", "bilinear")

    def __init__(
        self,
        rotation_range=0.0,
        zoom_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        shear_range=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        bounding_box_format=None,
        data_format=None,
        **kwargs,
    ):
        super().__init__(data_format=data_format, **kwargs)
        
        self.rotation_range = get_range(rotation_range,'rotation_range')
        self.zoom_range = get_range(zoom_range,'zoom_range')
        self.width_shift_range = get_range(width_shift_range,'width_shift_range')
        self.height_shift_range = get_range(height_shift_range,'height_shift_range')
        self.shear_range = get_range(shear_range,'shear_range')
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        self.seed = seed
        self.generator = SeedGenerator(seed)
        self.fill_mode = fill_mode
        self.interpolation = interpolation
        self.fill_value = fill_value
        self.supports_jit = False
        self.bounding_box_format = bounding_box_format
        
        if self.fill_mode not in self._SUPPORTED_FILL_MODE:
            raise NotImplementedError(
                f"Unknown `fill_mode` {fill_mode}. Expected of one "
                f"{self._SUPPORTED_FILL_MODE}."
            )
        if self.interpolation not in self._SUPPORTED_INTERPOLATION:
            raise NotImplementedError(
                f"Unknown `interpolation` {interpolation}. Expected of one "
                f"{self._SUPPORTED_INTERPOLATION}."
            )
    
    def get_config(self):
        config = {
            "rotation_range": self.rotation_range,
            "zoom_range": self.zoom_range,
            "width_shift_range": self.width_shift_range,
            "height_shift_range": self.height_shift_range,
            "shear_range": self.shear_range,
            "horizontal_flip": self.horizontal_flip,
            "vertical_flip": self.vertical_flip,
            "data_format": self.data_format,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "interpolation": self.interpolation,
            "seed": self.seed,
            "bounding_box_format": self.bounding_box_format,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape
    
    
        
    def _get_shape(self, images):
        shape = self.backbone.core.shape(images)
        if len(shape) == 4:
            batch_size = shape[0]
            if self.data_format == "channels_last":
                image_height = shape[1]
                image_width = shape[2]
            else:
                image_height = shape[2]
                image_width = shape[3]
        else:
            batch_size = 1
            if self.data_format == "channels_last":
                image_height = shape[0]
                image_width = shape[1]
            else:
                image_height = shape[1]
                image_width = shape[2]
        return batch_size, image_height, image_width

    def _get_t(self, images, transformations):
        batch_size, img_hd, img_wd = self._get_shape(images)
        A = get_affine_transform(self.backbone.cast(img_hd, self.backbone.float32),
                                 self.backbone.cast(img_wd, self.backbone.float32), **transformations)
        tt = self.backbone.stack(
            values=[
                A[..., 0, 0], A[..., 0, 1], A[..., 0, 2],
                A[..., 1, 0], A[..., 1, 1], A[..., 1, 2],
                A[..., 2, 0], A[..., 2, 1],
            ],
            axis=-1,
        )
        
        return tt
        
    def get_random_transformation(self, data, training=True, seed=None):
        ops = self.backend
        if not training:
            return None
        if isinstance(data, dict):
            images = data["images"]
        else:
            images = data
        batch_size, image_height, image_width = self._get_shape(images)
        
        if seed is None:
            seed = self._get_seed_generator(ops._backend)
        if self.rotation_range:
            theta = ops.random.uniform(
                shape=(batch_size,),
                minval=self.rotation_range[0], maxval=self.rotation_range[1],
                seed=seed,
            )
        else:
            theta = ops.zeros(batch_size, ops.float32)

        if self.height_shift_range:
            ty = ops.random.uniform(
                shape=(batch_size,),
                minval=self.height_shift_range[0], maxval=self.height_shift_range[1],
                seed=seed,
            )
        else:
            ty = ops.zeros(batch_size, ops.float32)

        if self.width_shift_range:
            tx = ops.random.uniform(
                shape=(batch_size,),
                minval=self.width_shift_range[0], maxval=self.width_shift_range[1],
                seed=seed,
            )
        else:
            tx = ops.zeros(batch_size, ops.float32)

        if self.shear_range:
            shear = ops.random.uniform(
                shape=(batch_size,),
                minval=self.shear_range[0], maxval=self.shear_range[1],
                seed=seed,
            )
        else:
            shear = tf.zeros(batch_size, tf.float32)

        if self.zoom_range:
            zx = ops.random.uniform(
                shape=(batch_size,),
                minval=1+self.zoom_range[0], maxval=1+self.zoom_range[1],
                seed=seed,
            )
            zy = ops.random.uniform(
                shape=(batch_size,),
                minval=1+self.zoom_range[0], maxval=1+self.zoom_range[1],
                seed=seed,
            )
        else:
            zx = ops.ones(batch_size, ops.float32)
            zy = ops.ones(batch_size, ops.float32) 
        
        if self.horizontal_flip:
            zx = ops.sign(ops.random.uniform(shape=(batch_size,), minval=-1, maxval=1)) * zx
        
        if self.vertical_flip:
            zy = ops.sign(ops.random.uniform(shape=(batch_size,), minval=-1, maxval=1)) * zy
            
        transformations = {
            "theta": theta,
            "tx": tx,
            "ty": ty,
            "shear": shear,
            "zx": zx,
            "zy": zy,
            "image_height": image_height,
            "image_width": image_width,
        }

        return transformations

    def transform_images(self, images, transformation, training=True):
        images = self.backend.cast(images, self.compute_dtype)
        if training:
            return self.backend.image.affine_transform(
                images=images,
                transform=_get_t(images, transformation),
                interpolation=self.interpolation,
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
                data_format=self.data_format,
            )
        return images
    
    def transform_segmentation_masks(self, segmentation_masks, transformation, training=True):
        return self.transform_images(
            segmentation_masks, transformation, training=training
        )
    
    def transform_bounding_boxes(self, bounding_boxes, transformation, training=True):
        if training:
            ops = self.backend
            height = transformation["image_height"]
            width = transformation["image_width"]
            
            bounding_boxes = converters.convert_format(
                bounding_boxes,
                source=self.bounding_box_format,
                target="xyxy",
                height=height,
                width=width,
            )

            A = get_affine_transform(height, width, **transformations)
            
            boxes = bounding_boxes["boxes"]
            ones = ops.ones_like(boxes[..., 0])
            point = ops.stack(
                [
                    ops.stack([boxes[..., 0], boxes[..., 1], ones], axis=-1), 
                    ops.stack([boxes[..., 2], boxes[..., 3], ones], axis=-1),
                    ops.stack([boxes[..., 0], boxes[..., 3], ones], axis=-1),
                    ops.stack([boxes[..., 2], boxes[..., 1], ones], axis=-1),
                ],
                axis=-1,
            )
        
            A = ops.linalg.pinv(A)
            out = ops.linalg.matmul(A[:, None, :, :], point)
            out = out[...,:2,:] / out[...,2:3,:]
            
            # find readjusted coordinates of bounding box to represent it in corners
            # format
            min_cordinates = ops.math.reduce_min(out, axis=-1)
            max_cordinates = ops.math.reduce_max(out, axis=-1)
            boxes = ops.concat([min_cordinates, max_cordinates], axis=-1)

            bounding_boxes = bounding_boxes.copy()
            bounding_boxes["boxes"] = boxes
            bounding_boxes = converters.clip_to_image_size(
                bounding_boxes,
                height=height,
                width=width,
                bounding_box_format="xyxy",
            )
            bounding_boxes = converters.convert_format(
                bounding_boxes,
                source="xyxy",
                target=self.bounding_box_format,
                height=height,
                width=width,
            )

        return bounding_boxes

    def transform_labels(self, labels, transformation, training=True):
        return labels

    
def get_translation_matrix(x, y):
    """Returns transform matrix(s) for the given translation(s).
    Returns:
      A tensor of shape `(..., 3, 3)`.
    """
    
    ones = tf.ones_like(x)
    zeros = tf.zeros_like(x)
    return tf.stack(
        values=[
            tf.stack(values=[ ones, zeros,    -x], axis = -1),
            tf.stack(values=[zeros,  ones,    -y], axis = -1),
            tf.stack(values=[zeros, zeros,  ones], axis = -1),
        ], axis=-2)

def get_rotation_matrix(theta):
    """Returns transform matrix(s) for given angle(s).
    Returns:
      A tensor of shape `(..., 3, 3)`.
    """
    theta = theta * np.pi / 180
    ones = tf.ones_like(theta)
    zeros = tf.zeros_like(theta)
    cos = tf.math.cos(theta)
    sin = tf.math.sin(theta)
    return tf.stack(
        values=[
            tf.stack(values=[  cos,  -sin, zeros], axis = -1),
            tf.stack(values=[  sin,   cos, zeros], axis = -1),
            tf.stack(values=[zeros, zeros,  ones], axis = -1),
        ], axis=-2)

def get_shear_matrix(shear):
    """Build ransform matrix(s) for given shear(s).
    Returns:
      A tensor of shape `(..., 3, 3)`
    """
    shear = shear * np.pi / 180
    ones = tf.ones_like(shear)
    zeros = tf.zeros_like(shear)
    cos = tf.math.cos(shear)
    sin = tf.math.sin(shear)
    return tf.stack(
        values=[
            tf.stack(values=[ ones,  -sin, zeros], axis = -1),
            tf.stack(values=[zeros,   cos, zeros], axis = -1),
            tf.stack(values=[zeros, zeros,  ones], axis = -1),
        ], axis=-2)

def get_zoom_matrix(zx, zy):
    """Build transform matrix(s) for given zoom.
    Returns:
      A tensor of shape `(..., 3, 3)`
    """
    ones = tf.ones_like(zx)
    zeros = tf.zeros_like(zx)
    return tf.stack(
        values=[
            tf.stack(values=[1./zx, zeros, zeros], axis = -1),
            tf.stack(values=[zeros, 1./zy, zeros], axis = -1),
            tf.stack(values=[zeros, zeros,  ones], axis = -1),
        ], axis=-2)

def get_affine_transform(
    img_hd, img_wd,
    theta,
    tx,
    ty,
    shear,
    zx,
    zy,
    name=None
):
    with tf.name_scope(name or "translation_matrix"):
        o_x = img_wd / 2.0
        o_y = img_hd / 2.0
        
        transform_matrix = get_translation_matrix(-o_x, -o_y)[None, ...]
        transform_matrix = tf.linalg.matmul(transform_matrix, get_rotation_matrix(theta))
        transform_matrix = tf.linalg.matmul(transform_matrix, get_translation_matrix(img_wd*tx, img_hd*ty))
        transform_matrix = tf.linalg.matmul(transform_matrix, get_shear_matrix(shear))
        transform_matrix = tf.linalg.matmul(transform_matrix, get_zoom_matrix(zx, zy))
        transform_matrix = tf.linalg.matmul(transform_matrix, get_translation_matrix(o_x, o_y)[None, ...])
        
        return transform_matrix


