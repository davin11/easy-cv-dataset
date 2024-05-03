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

import copy
from keras_cv.models import Task
from keras_cv.models import Backbone
import keras

def get_tensor_input_name(tensor):
    # keras_cv/models/utils.py
    try:
        return tensor._keras_history.operation.name
    except:
        return tensor.node.layer.name

def parse_model_inputs(input_shape, input_tensor, **kwargs):
    # keras_cv/models/utils.py
    if input_tensor is None:
        return keras.layers.Input(shape=input_shape, **kwargs)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            return keras.layers.Input(
                tensor=input_tensor, shape=input_shape, **kwargs
            )
        else:
            return input_tensor

def get_feature_extractor(model, layer_names, output_keys=None):
    """Create a feature extractor model with augmented output.
    from keras_cv.utils.train
    """
    if not output_keys:
        output_keys = layer_names
    items = zip(output_keys, layer_names)
    outputs = {key: model.get_layer(name).output for key, name in items}
    return keras.Model(inputs=model.inputs, outputs=outputs)

def Conv3x3BnReLU(filters, name, kernel_initializer = 'he_normal', use_batchnorm=False):
    def block(x):
        x = keras.layers.Conv2D(
                name=name+"_conv",
                filters=filters,
                kernel_size=(3,3),
                padding="same",
                use_bias=False,
                kernel_initializer=kernel_initializer
            )(x)
        if use_batchnorm:
            x = keras.layers.BatchNormalization(name=name+"_norm")(x)
        x = keras.layers.ReLU(name=name+"_relu")(x)
        return x
    return block

def DecoderUpsampling(filters, name, use_batchnorm=False):
    axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1

    def block(x, skip=None):
        x = keras.layers.UpSampling2D(size=(2,2), name=name+'_upsampling')(x)  # Nearest-neighbor interpolation
        x = keras.layers.Conv2D(filters, (2,2), padding='same', kernel_initializer='he_normal', name=name+'_lin')(x)
        if skip is not None:
            x = keras.layers.Concatenate(axis=axis, name=name+'_concat')([skip, x])
        x = Conv3x3BnReLU(filters, use_batchnorm=use_batchnorm, name=name+'_block1')(x)
        x = Conv3x3BnReLU(filters, use_batchnorm=use_batchnorm, name=name+'_block2')(x)
        return x

    return block


class UnetBackbone(Backbone):
    """Instantiates the U-Net Encoder architecture.

    References:
        - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597)

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        levels: int, number of levels, defaults to 4.
        filters: int, number of filters in the first level, defaults to 64.
        use_batchnorm: bool, whether to apply the BatchNormalization.

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # backbone
    model = UnetBackbone()
    output = model(input_data)
    ```
    """

    def __init__(
        self,
        include_rescaling=True,
        levels=4,
        filters=64,
        use_batchnorm=False,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        inputs = parse_model_inputs(input_shape, input_tensor)
        x = inputs

        if include_rescaling:
            x = keras.layers.Rescaling(1 / 255.0)(x)

        pyramid_level_inputs = dict()
        for l in range(levels):
            fn = (2**l) * filters
            if l>0:
                x = keras.layers.MaxPooling2D((2, 2), name=f'encoder{l-1}_pool')(x)
            x = Conv3x3BnReLU(fn, use_batchnorm=use_batchnorm, name=f'encoder{l}_block1')(x)
            x = Conv3x3BnReLU(fn, use_batchnorm=use_batchnorm, name=f'encoder{l}_block2')(x)
            pyramid_level_inputs[f"P{l}"] = get_tensor_input_name(x)

        x = keras.layers.MaxPooling2D((2, 2), name=f'encoder{levels-1}_pool')(x)
        x = Conv3x3BnReLU(16 * filters, use_batchnorm=use_batchnorm, name='center_block1')(x)
        x = keras.layers.Dropout(0.5)(x)
        x = Conv3x3BnReLU(16 * filters, use_batchnorm=use_batchnorm, name='center_block2')(x)
        pyramid_level_inputs[f"P{levels}"] = get_tensor_input_name(x)

        # Create model.
        super().__init__(inputs=inputs, outputs=x, **kwargs)

        # All references to `self` below this line
        self.pyramid_level_inputs = pyramid_level_inputs
        self.levels = levels
        self.filters = filters
        self.use_batchnorm = use_batchnorm

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "levels": self.levels,
                "filters": self.filters,
                "use_batchnorm": self.use_batchnorm,
            }
        )
        return config


class Unet(Task):
    """A Keras model implementing the U-Net a fully convolution neural network for image semantic segmentation

    References:
        - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597)

    Args:
        backbone: `keras.Model`. The backbone network for the model that is
            used as a feature extractor for the Encoder. Should
            either be a `keras_cv.models.Backbone`.
            A somewhat sensible backbone to use in many cases is the
            `keras_cv.models.ResNet50V2Backbone.from_preset("resnet50_v2_imagenet")`.
        num_classes: int, the number of classes for the detection model.
        filters: int, number of filters in the last level, defaults to 64.
        use_batchnorm: bool, whether to apply the BatchNormalization.

    Example:
    ```python
    import keras_cv

    images = np.ones(shape=(1, 224, 224, 3))
    labels = np.zeros(shape=(1, 224, 224, 1))
    backbone = keras_cv.models.ResNet50V2Backbone(input_shape=[96, 96, 3])
    model = Unet(num_classes=1, backbone=backbone)

    # Evaluate model
    model(images)

    # Train model
    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.fit(images, labels, epochs=3)
    ```
    """

    def __init__(
        self,
        num_classes,
        backbone=None,
        filters=64,
        use_batchnorm=False,
        **kwargs,
    ):

        if not isinstance(backbone, keras.layers.Layer) or not isinstance(
            backbone, keras.Model
        ):
            raise ValueError(
                "Argument `backbone` must be a `keras.layers.Layer` instance "
                f" or `keras.Model`. Received instead "
                f"backbone={backbone} (of type {type(backbone)})."
            )

        inputs = backbone.input

        extractor_levels = ["P0", "P1", "P2", "P3", "P4", "P5"]
        extractor_levels = [_ for _ in extractor_levels if _ in backbone.pyramid_level_inputs]
        extractor_layer_names = [
            backbone.pyramid_level_inputs[i] for i in extractor_levels
        ]
        feature_extractor = get_feature_extractor(
            backbone, extractor_layer_names, extractor_levels
        )

        backbone_features = feature_extractor(inputs)
        if "P5" in backbone_features:
            x = backbone_features["P5"]
            levels = 5
        elif "P4" in backbone_features:
            x = backbone_features["P4"]
            levels = 4
        elif "P3" in backbone_features:
            x = backbone_features["P3"]
            levels = 3
        else:
            raise ValueError(
                "Argument `backbone` must be a `keras.layers.Layer` instance "
                f" or `keras.Model`. Received instead "
                f"backbone={backbone} (of type {type(backbone)})."
            )

        for l in range(levels-1,-1,-1):
            fn = (2**l) * filters
            skip = backbone_features[f"P{l}"] if f"P{l}" in backbone_features else None
            x = DecoderUpsampling(fn, use_batchnorm=use_batchnorm, name=f'decoder{l}')(x, skip)

        if num_classes == 1:
            activation = 'sigmoid'
        else:
            activation = 'softmax'

        outputs = keras.layers.Conv2D(num_classes, (3, 3), activation=activation, padding='same',
                               kernel_initializer='he_normal')(x)

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        self.num_classes = num_classes
        self.backbone = backbone
        self.filters = filters
        self.use_batchnorm = use_batchnorm

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "backbone": keras.saving.serialize_keras_object(self.backbone),
            "filters": self.filters,
            "use_batchnorm": self.use_batchnorm,
        }

    @classmethod
    def from_config(cls, config):
        if "backbone" in config and isinstance(config["backbone"], dict):
            config["backbone"] = keras.layers.deserialize(config["backbone"])
        return super().from_config(config)
