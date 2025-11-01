import keras
from keras_hub.models import FeaturePyramidBackbone, Backbone, Task


def standardize_data_format(data_format):
    if data_format is None:
        return keras.config.image_data_format()
    data_format = str(data_format).lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            "{'channels_first', 'channels_last'}. "
            f"Received: data_format={data_format}"
        )
    return data_format

def Conv3x3BnReLU(filters, name, kernel_initializer = 'he_normal', use_batchnorm=False, data_format='channels_last'):
    axis = 3 if data_format == 'channels_last' else 1
    def block(x):
        x = keras.layers.Conv2D(
                name=name+"_conv",
                filters=filters,
                kernel_size=(3,3),
                padding="same",
                use_bias=False,
                kernel_initializer=kernel_initializer,
                data_format=data_format
            )(x)
        if use_batchnorm:
            x = keras.layers.BatchNormalization(name=name+"_norm", axis=axis)(x)
        x = keras.layers.ReLU(name=name+"_relu")(x)
        return x
    return block

def DecoderUpsampling(filters, name, use_batchnorm=False, data_format='channels_last'):
    axis = 3 if data_format == 'channels_last' else 1

    def block(x, skip=None):
        x = keras.layers.UpSampling2D(size=(2,2), name=name+'_upsampling', data_format=data_format)(x)  # Nearest-neighbor interpolation
        x = keras.layers.Conv2D(filters, (2,2), padding='same', kernel_initializer='he_normal',
                                name=name+'_lin', data_format=data_format)(x)
        if skip is not None:
            x = keras.layers.Concatenate(axis=axis, name=name+'_concat')([skip, x])
        x = Conv3x3BnReLU(filters, use_batchnorm=use_batchnorm, name=name+'_block1', data_format=data_format)(x)
        x = Conv3x3BnReLU(filters, use_batchnorm=use_batchnorm, name=name+'_block2', data_format=data_format)(x)
        return x

    return block

class UnetEncoder(FeaturePyramidBackbone):
    """Instantiates the U-Net Encoder architecture.

    References:
        - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597)

    Args:
        input_shape: shape tuple of input, defaults to (None, None, 3).
        levels: int, number of levels, defaults to 4.
        filters: int, number of filters in the first level, defaults to 64.
        use_batchnorm: bool, whether to apply the BatchNormalization.
        data_format: `None` or str. If specified, either `"channels_last"` or `"channels_first"`.

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # backbone
    model = UnetEncoder()
    output = model(input_data)
    ```
    """

    def __init__(
        self,
        levels=4,
        filters=64,
        use_batchnorm=False,
        input_shape=(None, None, 3),
        data_format=None,
        **kwargs,
    ):
        self._pyramid_outputs = {}
        data_format = standardize_data_format(data_format)
        inputs = keras.layers.Input(shape=input_shape)
        x = inputs

        for l in range(levels):
            fn = (2**l) * filters
            if l>0:
                x = keras.layers.MaxPooling2D((2, 2), name=f'encoder{l-1}_pool', data_format=data_format)(x)
            x = Conv3x3BnReLU(fn, use_batchnorm=use_batchnorm, name=f'encoder{l}_block1', data_format=data_format)(x)
            x = Conv3x3BnReLU(fn, use_batchnorm=use_batchnorm, name=f'encoder{l}_block2', data_format=data_format)(x)
            self._pyramid_outputs[f"P{l}"] = x

        x = keras.layers.MaxPooling2D((2, 2), name=f'encoder{levels-1}_pool', data_format=data_format)(x)
        x = Conv3x3BnReLU(16 * filters, use_batchnorm=use_batchnorm, name='center_block1', data_format=data_format)(x)
        x = keras.layers.Dropout(0.5)(x)
        x = Conv3x3BnReLU(16 * filters, use_batchnorm=use_batchnorm, name='center_block2', data_format=data_format)(x)
        self._pyramid_outputs[f"P{levels}"] = x

        # Create model.
        super().__init__(inputs=inputs, outputs=x, **kwargs)

        # All references to `self` below this line
        self.levels = levels
        self.filters = filters
        self.use_batchnorm = use_batchnorm
        self.data_format = data_format

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "levels": self.levels,
                "filters": self.filters,
                "use_batchnorm": self.use_batchnorm,
                "data_format": self.data_format,
            }
        )
        return config

class UnetBackbone(Backbone):
    """A Keras model implementing the U-Net a fully convolution neural network for image semantic segmentation

    References:
        - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597)

    Args:
        image_encoder: `keras_hub.models.FeaturePyramidBackbone`. The image encoder network for the model that is
            used as a feature extractor. If it is `None`, the classic U-Net encoder is used.
            A somewhat sensible backbone to use in many cases is the
            `keras_hub.models.Backbone.from_preset("resnet_50_imagenet")`.
        filters: int, number of filters in the last level, defaults to 64.
        use_batchnorm: bool, whether to apply the BatchNormalization.
        data_format: `None` or str. If specified, either `"channels_last"` or `"channels_first"`.

    Example:
    ```python
    import keras_cv

    images = np.ones(shape=(1, 224, 224, 3))
    labels = np.zeros(shape=(1, 224, 224, 1))
    image_encoder = keras_hub.models.Backbone.from_preset("resnet_50_imagenet")
    model = Unet(image_encoder=image_encoder)

    # Evaluate model
    model(images)

    ```
    """

    def __init__(
        self,
        image_encoder=None,
        filters=64,
        use_batchnorm=False,
        data_format=None,
        **kwargs,
    ):
        data_format = standardize_data_format(data_format)
        if image_encoder is None:
            image_encoder = UnetEncoder(filters=filters, data_format=data_format, use_batchnorm=use_batchnorm)
        assert data_format == image_encoder.data_format

        inputs = image_encoder.input
        extractor_levels = ["P0", "P1", "P2", "P3", "P4", "P5"]
        extractor_levels = [_ for _ in extractor_levels if _ in image_encoder.pyramid_outputs]
        backbone_features = {i: image_encoder.pyramid_outputs[i] for i in extractor_levels}
        
        if "P5" in extractor_levels:
            x = backbone_features["P5"]
            levels = 5
        if "P4" in extractor_levels:
            x = backbone_features["P4"]
            levels = 4
        elif "P3" in extractor_levels:
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
            x = DecoderUpsampling(fn, use_batchnorm=use_batchnorm, name=f'decoder{l}', data_format=data_format)(x, skip)

        super().__init__(inputs=inputs, outputs=x, **kwargs)

        # store metadata
        self.image_encoder = image_encoder
        self.filters = filters
        self.use_batchnorm = use_batchnorm
        self.data_format = data_format

    def get_config(self):
        base = super().get_config()
        base.update(
            {
                "image_encoder": keras.saving.serialize_keras_object(self.image_encoder),
                "filters": self.filters,
                "use_batchnorm": self.use_batchnorm,
                "data_format": self.data_format,
            }
        )
        return base

    @classmethod
    def from_config(cls, config):
        if "image_encoder" in config and isinstance(
            config["image_encoder"], dict
        ):
            config["image_encoder"] = keras.layers.deserialize(config["image_encoder"])
        return super().from_config(config)


class ImageSemanticSegmenter(Task):
    """A Keras model implementing for semantic segmentation."""

    def __init__(
        self,
        backbone,
        num_classes,
        preprocessor=None,
        activation=None,
        dropout=0.0,
        head_dtype=None,
        **kwargs,
    ):
        head_dtype = head_dtype or backbone.dtype_policy
        data_format = getattr(backbone, "data_format", None)

        # === Layers ===
        

        self.backbone = backbone
        self.preprocessor = preprocessor
        self.output_dropout = keras.layers.Dropout(
            dropout,
            dtype=head_dtype,
            name="output_dropout",
        )
        self.output_segmentation_head = keras.layers.Dense(
            num_classes,
            activation=activation,
            dtype=head_dtype,
            name="predictions",
        )

        # === Functional Model ===
        inputs = self.backbone.input
        x = self.backbone(inputs)
        x = self.output_dropout(x)
        outputs = self.output_segmentation_head(x)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.num_classes = num_classes
        self.activation = activation
        self.dropout = dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "backbone": keras.saving.serialize_keras_object(self.backbone),
            }
        )
        return config

    def get_config(self):
        # Backbone serialized in `super`
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "activation": self.activation,
                "dropout": self.dropout,
            }
        )
        return config

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        metrics="auto",
        **kwargs,
    ):
        """Configures the `ImageClassifier` task for training.

        The `ImageClassifier` task extends the default compilation signature of
        `keras.Model.compile` with defaults for `optimizer`, `loss`, and
        `metrics`. To override these defaults, pass any value
        to these arguments during compilation.

        Args:
            optimizer: `"auto"`, an optimizer name, or a `keras.Optimizer`
                instance. Defaults to `"auto"`, which uses the default optimizer
                for the given model and task. See `keras.Model.compile` and
                `keras.optimizers` for more info on possible `optimizer` values.
            loss: `"auto"`, a loss name, or a `keras.losses.Loss` instance.
                Defaults to `"auto"`, where a
                `keras.losses.SparseCategoricalCrossentropy` loss will be
                applied for the classification task. See
                `keras.Model.compile` and `keras.losses` for more info on
                possible `loss` values.
            metrics: `"auto"`, or a list of metrics to be evaluated by
                the model during training and testing. Defaults to `"auto"`,
                where a `keras.metrics.SparseCategoricalAccuracy` will be
                applied to track the accuracy of the model during training.
                See `keras.Model.compile` and `keras.metrics` for
                more info on possible `metrics` values.
            **kwargs: See `keras.Model.compile` for a full list of arguments
                supported by the compile method.
        """
        if optimizer == "auto":
            optimizer = keras.optimizers.Adam(5e-5)
        if loss == "auto":
            activation = getattr(self, "activation", None)
            activation = keras.activations.get(activation)
            from_logits = activation != keras.activations.softmax
            loss = keras.losses.CategoricalCrossentropy(from_logits)
        if metrics == "auto":
            metrics = [keras.metrics.CategoricalCrossentropy(),]
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs,
        )