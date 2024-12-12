import const
import keras
import numpy as np

@keras.saving.register_keras_serializable("Scale")
class Scale(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.f = self.add_weight(
            shape=(),
            initializer="random_normal",
            trainable=True,
            )
    
    def __call__(self, i):
        return  i * self.f
    
    def get_config(self):
        return super().get_config()

def main():
    prod = lambda p: p[0]*prod(p[1:]) if p else 1
    
    # Input
    i = keras.layers.Input(const.data_in+(1,))
    
    # Protien detection
    structure = (
            (5, 5, 2), # filters, kernel, strides
            (15, 5, 1),
            (20, 8, 1),
            (9, 9, 2)
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
    pd = keras.layers.Reshape((prod(pd.shape[1:-1]) , pd.shape[-1]))(pd)
    
    # Ability to sort
    ssize = (2,2,2)
    us = prod(ssize)
    d11 = keras.layers.Permute((2,1))(pd)
    d12 = keras.layers.Dense(
        6*const.num_predict_per,
        activation="leaky_relu",
        #kernel_regularizer="l2",
        #bias_regularizer="l2"
        )(d11)
    d13 = keras.layers.Dense(
        us*6*const.num_predict_per,
        activation="leaky_relu",
        #kernel_regularizer="l2",
        #bias_regularizer="l2"
        )(d12)
    op = d13
    for s in ssize:
        op = Scale()(op)
        op = keras.layers.MaxPooling1D(
            pool_size=s,
            data_format="channels_first",
            padding="valid",
            )(op)
    op = keras.layers.Flatten()(op)
    
    o1 = Scale()(op)
    o2 = keras.layers.Dense(
        3*6*const.num_predict_per,
        )(o1)
    o = o2
    model = keras.Model(inputs=i, outputs=o)
    
    model.compile(
            optimizer=keras.optimizers.LossScaleOptimizer(keras.optimizers.Nadam(
                learning_rate=keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=2e-4,
                        decay_steps=1e3,
                        decay_rate=0.96,
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

