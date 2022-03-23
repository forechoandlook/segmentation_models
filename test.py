import segmentation_models as sm  
from tensorflow import keras
# from segmentation_models import Backbones
from segmentation_models.models.unet import *
sm.set_framework('tf.keras')

sm.framework()


kwargs = {
    'backend': keras.backend,
    'layers': keras.layers,
    'models': keras.models,
    'utils': keras.utils,
}

backend, layers, models, keras_utils  = kwargs['backend'], kwargs['layers'], kwargs['models'], kwargs['utils']

set_kwargs(kwargs)

def build_unet_my(
        backbone,
        decoder_block,
        skip_connection_layers,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=5,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
):
    input_ = backbone.input
    x = backbone.output

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], layers.MaxPooling2D):
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block1')(x)
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block2')(x)

    # building decoder blocks
    last_feature = []
    for i in range(n_upsample_blocks):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)
        last_feature.append(upsampleN(stage=n_upsample_blocks-i)(x))

    # last feature upsample to initial image size and aggreate 

    last_feature = layers.Concatenate(axis=-1)(last_feature)
    x = Conv3x3BnReLU(decoder_filters[-1], use_batchnorm, name='last_feature_conv')(last_feature)

    # model head (define number of output classes)
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)
    x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    model = models.Model(input_, x)

    return model




def MyUnetOut(
        backbone_name='vgg16',
        input_shape=(None, None, 3),
        classes=1,
        activation='sigmoid',
        weights=None,
        encoder_weights='imagenet',
        encoder_freeze=False,
        use_attention = False,
        encoder_features='default',
        decoder_block_type='upsampling',
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
):

    backend, layers, models, keras_utils  = kwargs['backend'], kwargs['layers'], kwargs['models'], kwargs['utils']
    if decoder_block_type == 'upsampling':
        decoder_block = DecoderUpsamplingX2Block
    elif decoder_block_type == 'transpose':
        decoder_block = DecoderTransposeX2Block
    else:
        raise ValueError('Decoder block type should be in ("upsampling", "transpose"). '
                         'Got: {}'.format(decoder_block_type))

    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs,
    )
    if encoder_features == 'default':
        encoder_features = Backbones.get_feature_layers(backbone_name, n=4)
    
    if not use_attention:
        model = build_unet(
            backbone=backbone,
            decoder_block=decoder_block,
            skip_connection_layers=encoder_features,
            decoder_filters=decoder_filters,
            classes=classes,
            activation=activation,
            n_upsample_blocks=len(decoder_filters),
            use_batchnorm=decoder_use_batchnorm,
        )
    else:
        model = build_unet_my(
            backbone=backbone,
            decoder_block=decoder_block,
            skip_connection_layers=encoder_features,
            decoder_filters=decoder_filters,
            classes=classes,
            activation=activation,
            n_upsample_blocks=len(decoder_filters),
            use_batchnorm=decoder_use_batchnorm,
        )
    
    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model

model = MyUnetOut("resnet34",input_shape=(480, 320, 1), 
                    encoder_weights=None, classes=12, 
                    activation='softmax', use_attention=True)
print(model.summary())

# plot model architecture
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_my_att.png', show_shapes=True, show_layer_names=True)