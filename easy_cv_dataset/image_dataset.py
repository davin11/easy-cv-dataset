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
# ==============================================================================
"""Keras image dataset loading utilities."""


import os
import pandas
import numpy as np
import tensorflow as tf
from functools import partial
from PIL import Image
import keras
from .utils import dataframe_from_directory
from .utils import dataset_from_dataframe
from keras_cv import bounding_box
from keras_cv.layers import Rescaling

IMAGES = "images"
LABELS = "labels"
BOUNDING_BOXES = "bounding_boxes"
KEYPOINTS = "keypoints"
SEGMENTATION_MASKS = "segmentation_masks"
ALLOW_LIST_FORMATS = (".bmp", ".tiff", ".tif", ".jpeg", ".jpg", ".png")


def image_dataframe_from_directory(
    directory, follow_links=False
):
    """Generates a `pandas.DataFrame` from image files in a directory.

    If your directory structure is:

    ```
    main_directory/
    ...class_a/
    ......a_image_1.jpg
    ......a_image_2.jpg
    ...class_b/
    ......b_image_1.jpg
    ......b_image_2.jpg
    ```

    Then calling `image_dataframe_from_directory(main_directory)` will 
    return a `pandas.DataFrame` that yields batches of
    images from the subdirectories `class_a` and `class_b`.

    Supported image formats: jpeg, png, bmp, tiff.

    Args:
      directory: Directory where the data is located.
      follow_links: Whether to visit subdirectories pointed to by symlinks.
          Defaults to False.

    Returns:
      A `pandas.DataFrame` object.

    """
    return dataframe_from_directory(
        directory=directory,
        formats=ALLOW_LIST_FORMATS,
        colname_file="image",
        colname_class="class",
        follow_links=follow_links
    )


def _load_img(path, num_channels=3):
    """Load an image from a path and resize it."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(
        img, channels=num_channels, expand_animations=False
    )
    return img


def _load_map_np(path):
    if isinstance(path, np.ndarray):
        path = path.item()
    return np.asarray(Image.open(path.decode()), np.int64)


def _load_map(path, class_mode, num_classes):
    y = tf.numpy_function(_load_map_np, [path,], Tout=tf.int64)
    y.set_shape([None, None])
    if class_mode == "binary":
        y = tf.cast(y, tf.float32) / 255.0
        y = tf.expand_dims(y, -1)
    elif class_mode == "int":
        y = tf.cast(y, "int64")
    elif class_mode == "categorical":
        y = tf.one_hot(y, num_classes)
    elif class_mode == "raw":
        y = tf.cast(y, "float32")
    else:
        raise ValueError(
            "`class_mode` argument must be "
            'one of "int", "binary", "categorical", or "raw". '
            f"Received: class_mode={class_mode}"
        )
    return y


def _dict_to_tuple_fun(dat, dictname_input, dictname_target, max_boxes=None):
    x = dat[dictname_input]
    y = dat[dictname_target]
    if dictname_target == BOUNDING_BOXES:
        y = bounding_box.to_dense(y, max_boxes=max_boxes)
    return x, y


def _get_fun_load_class(x, class_mode, num_classes):
    if (not isinstance(class_mode, str)) and callable(class_mode):
        return class_mode(x)
    elif class_mode == "none":
        return x
    elif class_mode == "raw":
        return tf.cast(x, "float32")
    elif class_mode == "int":
        return tf.cast(x, "int64")
    elif class_mode == "categorical":
        return tf.one_hot(tf.cast(x, "int64"), num_classes)
    else:
        raise ValueError(
            "`class_mode` argument must be callable or "
            'one of "int", "categorical", or "raw". '
            f"Received: class_mode={class_mode}"
        )


def _update_post_batching_processing(post_batching_processing, do_normalization, dictname_input, dictname_target, max_boxes=None):
    to_tuple = partial(_dict_to_tuple_fun,
                       dictname_input=dictname_input,
                       dictname_target=dictname_target,
                       max_boxes=max_boxes
                       )

    if post_batching_processing is None:
        if do_normalization:
            post_batching_processing = keras.Sequential(
                layers=[Rescaling(1 / 255.0), to_tuple]
            )
        else:
            post_batching_processing = to_tuple
    elif do_normalization is None:
        post_batching_processing = keras.Sequential(
            layers=[post_batching_processing, Rescaling(1 / 255.0), to_tuple]
        )
    else:
        post_batching_processing = keras.Sequential(
            layers=[post_batching_processing, to_tuple]
        )
    return post_batching_processing


def image_classification_dataset_from_dataframe(
    dataframe,
    root_path=None,
    shuffle=True,
    seed=None,
    class_names=None,
    class_mode="categorical",
    color_mode="rgb",
    pre_batching_processing=None,
    batch_size=None,
    post_batching_processing=None,
    do_normalization=False,
    colname_image="image",
    colname_class="class",
):
    """Generates a `tf.data.Dataset` from a dataframe.

    Then calling `image_dataset_from_dataframe(dataframe)`
    will return a `tf.data.Dataset`.

    Args:
      dataframe: Dataframe with list of images.
      root_path: if it is not None, all paths are relative to it.
      shuffle: Whether to shuffle the data. Default: True.
          If set to False, sorts the data in alphanumeric order.
      seed: Optional random seed for shuffling and transformations.
      class_mode: String describing the encoding of classes. Options are:
          - 'raw': means that the classes are not encoded
          - 'int': means that the classes are encoded as integers
              (e.g. for `sparse_categorical_crossentropy` loss).
          - 'categorical' means that the classes are
              encoded as a categorical vector
              (e.g. for `categorical_crossentropy` loss).
          - None (no labels).
      class_names: Only valid if "class_mode" is 'int' or 'categorical'. This is the explicit
          list of class names used for the encoding.
          (otherwise alphanumeric order is used).
      color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
          Whether the images will be converted to
          have 1, 3, or 4 channels.
      pre_batching_processing: The operation(s) to apply before the data is put into a batch.
      batch_size: Size of the batches of data. Default: None.
        If `None`, the data will not be batched
        (the dataset will yield individual samples).
      post_batching_processing: The operation(s) to apply after tha data has been batched.
      do_normalization: If True, normalize the image in the range [0,1].
      colname_image: column name for image file. Default: 'image'.
      colname_class:: column name for class. Default: 'class'.
    Returns:
      A `tf.data.Dataset` object.

    Rules regarding labels format:

      - if `class_mode` is `int`, the labels are an `int32` tensor of shape `(batch_size,)`.
      - if `class_mode` is `raw`, the classes are not converted.
      - if `class_mode` is `categorical`, the labels are a `float32` tensor
        of shape `(batch_size, num_classes)`, representing a one-hot
        encoding of the class index.

    Rules regarding number of channels in the yielded images:

      - if `color_mode` is `grayscale`,
        there's 1 channel in the image tensors.
      - if `color_mode` is `rgb`,
        there are 3 channels in the image tensors.
      - if `color_mode` is `rgba`,
        there are 4 channels in the image tensors.
    """
    if seed is None:
        seed = np.random.randint(1e6)

    if isinstance(dataframe, str):
        dataframe = pandas.read_csv(dataframe)
    dataframe = dataframe.copy()

    if root_path is not None:
        dataframe[colname_image] = [
            os.path.join(root_path, _) for _ in dataframe[colname_image]
        ]

    num_classes = 0
    if class_mode in {"int", "categorical"}:
        if class_names is None:
            class_names = sorted(list(set(dataframe[colname_class])))

        dataframe = dataframe[dataframe[colname_class].isin(class_names)]
        dataframe[colname_class] = [
            class_names.index(_) for _ in dataframe[colname_class]
        ]
        num_classes = len(class_names)
        for index_class, name_class in enumerate(class_names):
            num_img = np.sum(dataframe[colname_class] == index_class)
            print(
                f"For class '{name_class}', there are {num_img} images.",
                flush=True,
            )

    elif class_names:
        raise ValueError(
            "You can only pass `class_names` if "
            '`class_mode` is "int" or "categorical".'
            f"Received: class_mode={class_mode}, and "
            f"class_names={class_names}"
        )

    if (not isinstance(class_mode, str)) and callable(class_mode):
        load_fun_target = class_mode
    else:
        load_fun_target = partial(
            _get_fun_load_class, class_mode=class_mode, num_classes=num_classes
        )

    if color_mode == "rgb":
        num_channels = 3
    elif color_mode == "rgba":
        num_channels = 4
    elif color_mode == "grayscale":
        num_channels = 1
    else:
        raise ValueError(
            '`color_mode` must be one of {"rgb", "rgba", "grayscale"}. '
            f"Received: color_mode={color_mode}"
        )

    load_fun_input = partial(_load_img, num_channels=num_channels)
    post_batching_processing = _update_post_batching_processing(
        post_batching_processing=post_batching_processing,
        do_normalization=do_normalization,
        dictname_input=dictname_input,
        dictname_target=dictname_target,
        max_boxes=None)

    dataset = dataset_from_dataframe(
        dataframe=dataframe,
        colname_input=colname_image,
        colname_target=colname_class,
        load_fun_input=load_fun_input,
        load_fun_target=load_fun_target,
        shuffle=shuffle,
        seed=seed,
        pre_batching_operation=pre_batching_processing,
        batch_size=batch_size,
        post_batching_operation=post_batching_processing,
        dictname_input=IMAGES,
        dictname_target=LABELS,
    )

    # Users may need to reference `class_names`, `batch_size` and `root_path`
    dataset.class_names = class_names
    dataset.batch_size = batch_size
    dataset.root_path = root_path
    return dataset


def image_segmentation_dataset_from_dataframe(
    dataframe,
    class_names,
    root_path=None,
    shuffle=True,
    seed=None,
    class_mode="categorical",
    color_mode="rgb",
    pre_batching_processing=None,
    batch_size=None,
    post_batching_processing=None,
    do_normalization=False,
    colname_image="image",
    colname_mask="segmentation_mask",
):
    """Generates a `tf.data.Dataset` from a dataframe.

    Calling `image_segmentation_dataset_from_dataframe(dataframe)`
    will return a `tf.data.Dataset` that yields batches of
    images from the subdirectories `class_a` and `class_b`, together with labels
    0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

    Args:
      dataframe: Dataframe with list of images.
      root_path: if it is not None, all paths are relative to it.
      shuffle: Whether to shuffle the data. Default: True.
          If set to False, sorts the data in alphanumeric order.
      seed: Optional random seed for shuffling and transformations.
      class_mode: String describing the encoding of classes. Options are:
          - 'int': means that the classes are encoded as integers
              (e.g. for `sparse_categorical_crossentropy` loss).
          - 'categorical' means that the classes are
              encoded as a categorical vector
              (e.g. for `categorical_crossentropy` loss).
      class_names: Only valid if "class_mode" is 'int' or 'categorical'. This is the explicit
          list of class names used for the encoding.
          (otherwise alphanumeric order is used).
      color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
          Whether the images will be converted to
          have 1, 3, or 4 channels.
      pre_batching_processing: The operation(s) to apply before the data is put into a batch.
      batch_size: Size of the batches of data. Default: None.
        If `None`, the data will not be batched
        (the dataset will yield individual samples).
      post_batching_processing: The operation(s) to apply after tha data has been batched.
      do_normalization: If True, normalize the image in the range [0,1].
      colname_image: column name for image file. Default: 'image'.
      colname_mask: column name for mask file. Default: 'segmentation_mask'.
    Returns:
      A `tf.data.Dataset` object.

    Rules regarding labels format:

      - if `class_mode` is `int`, the labels are an `int32` tensor of shape
        `(batch_size,)`.
      - if `class_mode` is `raw`, the classes are converted.
      - if `class_mode` is `categorical`, the labels are a `float32` tensor
        of shape `(batch_size, num_classes)`, representing a one-hot
        encoding of the class index.

    Rules regarding number of channels in the yielded images:

      - if `color_mode` is `grayscale`,
        there's 1 channel in the image tensors.
      - if `color_mode` is `rgb`,
        there are 3 channels in the image tensors.
      - if `color_mode` is `rgba`,
        there are 4 channels in the image tensors.
    """
    if seed is None:
        seed = np.random.randint(1e6)

    if isinstance(dataframe, str):
        dataframe = pandas.read_csv(dataframe)
    dataframe = dataframe.copy()

    if root_path is not None:
        dataframe[colname_image] = [
            os.path.join(root_path, _) for _ in dataframe[colname_image]
        ]
        dataframe[colname_mask] = [
            os.path.join(root_path, _) for _ in dataframe[colname_mask]
        ]

    num_classes = len(class_names)
    load_fun_target = partial(
        _load_map, class_mode=class_mode, num_classes=num_classes
    )

    if color_mode == "rgb":
        num_channels = 3
    elif color_mode == "rgba":
        num_channels = 4
    elif color_mode == "grayscale":
        num_channels = 1
    else:
        raise ValueError(
            '`color_mode` must be one of {"rgb", "rgba", "grayscale"}. '
            f"Received: color_mode={color_mode}"
        )

    load_fun_input = partial(_load_img, num_channels=num_channels)
    post_batching_processing = _update_post_batching_processing(
        post_batching_processing=post_batching_processing,
        do_normalization=do_normalization,
        dictname_input=dictname_input,
        dictname_target=dictname_target,
        max_boxes=None)

    dataset = dataset_from_dataframe(
        dataframe=dataframe,
        colname_input=colname_image,
        colname_target=colname_mask,
        load_fun_input=load_fun_input,
        load_fun_target=load_fun_target,
        shuffle=shuffle,
        seed=seed,
        pre_batching_operation=pre_batching_processing,
        batch_size=batch_size,
        post_batching_operation=post_batching_processing,
        dictname_input=IMAGES,
        dictname_target=SEGMENTATION_MASKS,
    )

    # Users may need to reference `class_names`, `batch_size` and `root_path`
    dataset.class_names = class_names
    dataset.batch_size = batch_size
    dataset.root_path = root_path
    return dataset


def _get_objdetect_generator(
    dataframe, colname_image, colname_class, colname_box, class_mode, num_classes
):
    list_img = list(set(dataframe[colname_image]))
    for image in list_img:
        sel = dataframe[dataframe[colname_image] == image]
        yield {
            IMAGES: image,
            BOUNDING_BOXES: {
                "boxes": tf.cast(sel[colname_box].values, "float32"),
                "classes": _get_fun_load_class(sel[colname_class].values, class_mode=class_mode, num_classes=num_classes),
            },
        }



def image_objdetect_dataset_from_dataframe(
    dataframe,
    root_path=None,
    shuffle=True,
    seed=None,
    class_names=None,
    class_mode="int",
    color_mode="rgb",
    pre_batching_processing=None,
    batch_size=None,
    post_batching_processing=None,
    do_normalization=False,
    colname_image="image",
    colname_class="class",
    colname_box=["xmin", "ymin", "xmax", "ymax"],
    bounding_box_input_format="xyxy",
    bounding_box_format="xyxy",
    max_boxes=None,
):
    """Generates a `tf.data.Dataset` from a dataframe.

    Calling `image_objdetect_dataset_from_dataframe(dataframe)`
    will return a `tf.data.Dataset` that yields batches of
    images from the subdirectories `class_a` and `class_b`, together with labels
    0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

    Args:
      dataframe: Dataframe with list of images.
      root_path: if it is not None, all paths are relative to it.
      shuffle: Whether to shuffle the data. Default: True.
          If set to False, sorts the data in alphanumeric order.
      seed: Optional random seed for shuffling and transformations.
      class_mode: String describing the encoding of classes. Options are:
          - 'raw': means that the classes are not encoded
          - 'int': means that the classes are encoded as integers
              (e.g. for `sparse_categorical_crossentropy` loss).
          - 'categorical' means that the classes are
              encoded as a categorical vector
              (e.g. for `categorical_crossentropy` loss).
          - None (no labels).
      class_names: Only valid if "class_mode" is 'int' or 'categorical'. This is the explicit
          list of class names used for the encoding.
          (otherwise alphanumeric order is used).
      color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
          Whether the images will be converted to
          have 1, 3, or 4 channels.
      pre_batching_processing: The operation(s) to apply before the data is put into a batch.
      batch_size: Size of the batches of data. Default: 32.
        If `None`, the data will not be batched
        (the dataset will yield individual samples).
      post_batching_processing: The operation(s) to apply after tha data has been batched.
      do_normalization: If True, normalize the image in the range [0,1].
      colname_image: column name for image file. Default: 'image'.
      colname_class: column name for class. Default: 'class'.
      colname_box: column names for box position. Default: ["xmin", "ymin", "xmax", "ymax"].
    Returns:
      A `tf.data.Dataset` object.

    Rules regarding labels format:

      - if `class_mode` is `int`, the labels are an `int32` tensor of shape
        `(batch_size,)`.
      - if `class_mode` is `raw`, the classes are converted.
      - if `class_mode` is `categorical`, the labels are a `float32` tensor
        of shape `(batch_size, num_classes)`, representing a one-hot
        encoding of the class index.

    Rules regarding number of channels in the yielded images:

      - if `color_mode` is `grayscale`,
        there's 1 channel in the image tensors.
      - if `color_mode` is `rgb`,
        there are 3 channels in the image tensors.
      - if `color_mode` is `rgba`,
        there are 4 channels in the image tensors.
    """

    if isinstance(dataframe, str):
        dataframe = pandas.read_csv(dataframe)
    dataframe = dataframe.copy()

    if root_path is not None:
        dataframe[colname_image] = [
            os.path.join(root_path, _) for _ in dataframe[colname_image]
        ]

    num_classes = 0
    if class_mode in {"int", "categorical"}:
        if class_names is None:
            class_names = sorted(list(set(dataframe[colname_class])))

        dataframe = dataframe[dataframe[colname_class].isin(class_names)]
        dataframe[colname_class] = [
            class_names.index(_) for _ in dataframe[colname_class]
        ]
        num_classes = len(class_names)
        for index_class, name_class in enumerate(class_names):
            num_boxes = np.sum(dataframe[colname_class] == index_class)
            print(
                f"For class '{name_class}', there are {num_boxes} boxes.",
                flush=True,
            )
    elif class_names:
        raise ValueError(
            "You can only pass `class_names` if "
            '`class_mode` is "int" or "categorical".'
            f"Received: class_mode={class_mode}, and "
            f"class_names={class_names}"
        )

    if color_mode == "rgb":
        num_channels = 3
    elif color_mode == "rgba":
        num_channels = 4
    elif color_mode == "grayscale":
        num_channels = 1
    else:
        raise ValueError(
            '`color_mode` must be one of {"rgb", "rgba", "grayscale"}. '
            f"Received: color_mode={color_mode}"
        )

    load_fun_input = partial(_load_img, num_channels=num_channels)
    load_fun_target = partial(
        bounding_box.convert_format,
        source=bounding_box_input_format,
        target=bounding_box_format,
        dtype="float32",
    )
    
    post_batching_processing = _update_post_batching_processing(
        post_batching_processing=post_batching_processing,
        do_normalization=do_normalization,
        dictname_input=dictname_input,
        dictname_target=dictname_target,
        max_boxes=max_boxes)

    if shuffle:
        if seed is None:
            seed = np.random.randint(1e6)
        dataframe = dataframe.sample(frac=1, random_state=seed)

    num_img = len(set(dataframe[colname_image]))
    class_dtype = "int64" if class_mode in {"int", "categorical"} else "float32"
    class_shape = (None,num_classes) if class_mode in {"categorical", } else (None,)

    generator = partial(
        _get_objdetect_generator,
        dataframe=dataframe,
        colname_image=colname_image,
        colname_class=colname_class,
        colname_box=colname_box,
        class_mode=class_mode,
        num_classes=num_classes,
    )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature={
            IMAGES: tf.TensorSpec(shape=(), dtype="string"),
            BOUNDING_BOXES: {
                "boxes": tf.TensorSpec(shape=(None, 4), dtype="float32"),
                "classes": tf.TensorSpec(shape=class_shape, dtype=class_dtype),
            },
        },
    )
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(num_img))

    dataset = dataset_from_dataframe(
        dataframe=dataset,
        colname_input=IMAGES,
        colname_target=BOUNDING_BOXES,
        load_fun_input=load_fun_input,
        load_fun_target=load_fun_target,
        shuffle=shuffle,
        seed=seed,
        pre_batching_operation=pre_batching_processing,
        batch_size=batch_size,
        post_batching_operation=post_batching_processing,
        dictname_input=IMAGES,
        dictname_target=BOUNDING_BOXES,
    )

    # Users may need to reference `class_names`, `batch_size` and `root_path`
    dataset.class_names = class_names
    dataset.batch_size = batch_size
    dataset.bounding_box_format = bounding_box_format
    dataset.root_path = root_path
    return dataset
