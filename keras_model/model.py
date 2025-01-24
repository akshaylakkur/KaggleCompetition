import const
import keras
from keras import ops
from keras.layers import Layer, Input, \
    Reshape, Permute, Concatenate, Dense, \
    Conv3D, MaxPooling3D, MaxPooling2D, UpSampling2D
from keras.losses import huber
import numpy as np

prod = lambda p: p[0]*prod(p[1:]) if p else 1

@keras.saving.register_keras_serializable("Influence")
class Influence(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inp):
        inp = ops.reshape(inp, inp.shape+(1,))
        tra = ops.swapaxes(inp, -1, -2)
        mul = inp @ tra
        s = mul.shape
        s = s[:-2] + (s[-1]*s[-2],)
        return ops.reshape(mul, s)
    
    def get_config(self):
        return super().get_config()

def main():
    # Input
    i = Input(const.data_in+(1,))
    
    # Protien detection
    structure = (
            (4, 7, 1, "leaky_relu"), # filters, kernel, strides, activation
            (4, 7, 1, "softmax"),
            (8, 7, 1, None),
            (12, 7, 1, None),
            (14, 7, 1, "leaky_relu")
            )
    pools = (
            (1,2,2),
            2,
            2,
            2,
            2,
            )
    pd = i
    for (filters, kernel, strides, activation), pool in zip(structure, pools):
        pd = Conv3D(
                filters=filters,
                kernel_size=kernel,
                strides=strides,
                padding="same",
                kernel_regularizer="l2",
                bias_regularizer="l1",
                )(pd)
        if activation in ('softmax', 'selu'):
            pd = keras.layers.BatchNormalization()(pd)
        pd = keras.layers.Activation(activation=activation)(pd)
        pd = MaxPooling3D(
            pool_size=pool,
            )(pd)
        if filters<10:
            pd = Influence()(pd)
    
    sh = pd.shape
    num = sh[1]*sh[2]*sh[3]
    op = keras.layers.Reshape((num, sh[-1]))(pd)
    op = keras.layers.Conv1D(
        filters=3,
        kernel_size=5,
        padding="same",
        activation="leaky_relu",
        kernel_regularizer="l1l2",
        bias_regularizer="l1",
        )(op)
    op = keras.layers.Permute((2,1))(op)
    op = keras.layers.Dense(100, activation="squareplus")(op)
    op = keras.layers.Permute((2,1))(op)
    
    o = op
    model = keras.Model(inputs=i, outputs=o)
    
    model.compile(
            optimizer=keras.optimizers.LossScaleOptimizer(keras.optimizers.Nadam(
                learning_rate=keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=7e-4,
                        decay_steps=1e2,
                        decay_rate=1e-3,
                        staircase=False
                        )
                )
                ),
            loss=loss,
            metrics=[
                "mae",
                "mse",
                ]
            )
    model.summary()
    return model

@keras.saving.register_keras_serializable("loss")
def loss(y_true, y_pred):
    ae = abs(y_pred-y_true)
    se = ae*ae
    
    ae = ae.mean()
    se = se.mean()
    return ae*se/1e7

