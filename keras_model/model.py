import const
import keras
import numpy as np

prod = lambda p: p[0]*prod(p[1:]) if p else 1

@keras.saving.register_keras_serializable("Mul")
class Mul(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, i):
        m1 = i[:, :, :, :, np.newaxis]
        m2 = i[:, :, :, np.newaxis, :]
        m = m1 @ m2
        return keras.layers.Reshape(m.shape[1:-2]+(m.shape[-1]*m.shape[-2],))(m)
    
    def get_config(self):
        return super().get_config()

@keras.saving.register_keras_serializable("Form")
class Form(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, i):
        compat = keras.layers.Reshape((1, prod(i.shape[1:-1]) , i.shape[-1]))(i)
        sized = keras.layers.UpSampling2D(
            size=(6,1),
            data_format="channels_last",
            )(compat)
        
        non_softmax = sized[:, :, :, :-6]
        to_softmax = sized[:, :, :, -6:]
        softmaxed = keras.ops.softmax(to_softmax)
        normed = keras.layers.Concatenate(axis=-1)([non_softmax, softmaxed])
        
        to_embed = keras.ops.pad(normed, ((0,0),)*3+((0,6),), constant_values=1)
        embedding = np.zeros(to_embed.shape[1:])
        for i in range(6):
            embedding[i, :, -6+i] = 1
        embedding[:, :, :-6] = 1
        print(to_embed.shape)
        print(embedding.shape)
        embedded = to_embed * embedding
        
        return embedded
    
    def get_config(self):
        return super().get_config()

def main():
    # Input
    i = keras.layers.Input(const.data_in+(1,))
    
    # Protien detection
    structure = (
            (5, 5, 2), # filters, kernel, strides
            (15, 15, 1),
            (10, 8, 1),
            (9, 6, 2)
            )
    pools = (
            (2,1,1),
            (1,2,2),
            (2,1,1),
            (1,2,2),
            )
    pd = i
    for (filters, kernel, strides), pool in zip(structure, pools):
        pd = keras.layers.Conv3D(
                filters=filters,
                kernel_size=kernel,
                strides=strides,
                padding="same",
                kernel_regularizer="l1",
                bias_regularizer="l1"
                )(pd)
        pd = keras.layers.MaxPooling3D(
            pool_size=pool,
            padding="valid",
            )(pd)
    
    op = Form()(pd) # Transition the data
    
    # Ability to sort
    ssize = (
        (5,3), # features, candidates
        (6,4),
        (3,4)
        )
    for f,s in ssize:
        op = Mul()(op)
        op = keras.layers.Dense(
            f,
            activation="linear",
            )(op)
        op = keras.layers.Permute((1,3,2))(op)
        op = keras.layers.Dense(
            s*const.num_predict_per,
            activation="linear",
            )(op)
        op = keras.layers.Permute((1,3,2))(op)
        op = keras.layers.MaxPooling2D(
            pool_size=(1,s),
            data_format="channels_last",
            padding="valid",
            )(op)
    
    o = op
    model = keras.Model(inputs=i, outputs=o)
    
    model.compile(
            optimizer=keras.optimizers.LossScaleOptimizer(keras.optimizers.Adamax(
                learning_rate=keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=2e-4,
                        decay_steps=1e3,
                        decay_rate=0.98,
                        staircase=False
                        )
                )
                ),
            loss="mse",
            metrics=[
                "mae",
                ]
            )
    model.summary()
    return model

