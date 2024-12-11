import const
import keras
import numpy as np

@keras.saving.register_keras_serializable("Format")
class Format(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, i):
        l = i[:, :, :3]
        
        r = i[:, :, 3:]
        r = keras.ops.sigmoid(r)
        r2 = 6*keras.ops.average(r, axis=-1)
        r = r / r2[:, :, np.newaxis]
        return keras.ops.concatenate((l, r), axis=-1)
    
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
            (3, 9, 2)
            )
    pools = (
            (2,1,1),
            (1,2,2),
            (2,2,2)
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
    
    d11 = keras.layers.Permute((2,1))(pd)
    d12 = keras.layers.Dense(
        6*const.num_predict_per,
        activation="leaky_relu",
        kernel_regularizer="l2",
        bias_regularizer="l2"
        )(d11)
    d13 = keras.layers.Dense(
        4*6*const.num_predict_per,
        activation="leaky_relu",
        kernel_regularizer="l2",
        bias_regularizer="l2"
        )(d12)
    d14 = keras.layers.MaxPooling1D(
        pool_size=2,
        data_format="channels_first",
        padding="valid",
        )(d13)
    d15 = keras.layers.Dense(
        2*6*const.num_predict_per,
        activation="leaky_relu",
        kernel_regularizer="l2",
        bias_regularizer="l2"
        )(d14)
    d16 = keras.layers.MaxPooling1D(
        pool_size=2,
        data_format="channels_first",
        padding="valid",
        )(d15)
    d17 = keras.layers.Dense(
        6*const.num_predict_per,
        activation="leaky_relu",
        kernel_regularizer="l2",
        bias_regularizer="l2"
        )(d16)
    d1 = d17
    
    o1 = keras.layers.Permute((2,1))(d1)
    #o2 = Format()(o1)
    o = o1
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
            loss="mse",#calibrated_loss,
            metrics=[
                "mae",
                #"mse",
                #mse_first,
                #cce_last,
                #joint_loss
                ]
            )
    model.summary()
    return model

@keras.saving.register_keras_serializable("calibrated_loss")
def calibrated_loss(y_true, y_pred):
    weight_mse = 1e-8
    weight_cce = 1-weight_mse
    
    wmse = mse_first(y_true, y_pred)
    wcce = cce_last(y_true, y_pred)
    return weight_mse*wmse + weight_cce*wcce

@keras.saving.register_keras_serializable("joint_loss")
def joint_loss(y_true, y_pred):
    wmse = mse_first(y_true, y_pred)
    wcce = cce_last(y_true, y_pred)
    return wmse * wcce

@keras.saving.register_keras_serializable("mse_first")
def mse_first(y_true, y_pred):
    yt = y_true[:, :, :3]
    yp = y_pred[:, :, :3]
    return ((yt-yp)**2).sum(axis=-1)

@keras.saving.register_keras_serializable("cce_last")
def cce_last(y_true, y_pred):
    yt = y_true[:, :, 3:]
    yp = y_pred[:, :, 3:]
    
    return keras.losses.CategoricalCrossentropy(
        from_logits=False,
        reduction=None
        )(yt, yp)
