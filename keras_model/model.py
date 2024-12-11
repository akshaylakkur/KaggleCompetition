import const
import keras
import numpy as np

def main():
    prod = lambda p: p[0]*prod(p[1:]) if p else 1
    
    # Input
    i = keras.layers.Input(const.data_in)
    i1 = keras.layers.Reshape(const.data_in+(1,))(i)
    
    # Protien detection
    structure = (
            (3, 8, (2,2,2)), # filters, kernel, strides
            (10, 8, (2,2,2)),
            (9, 8, 1)
            )
    pools = (
            (2,1,1),
            (2,3,3),
            (2,2,2)
            )
    pd = i1
    lys = []
    for (filters, kernel, strides), pool in zip(structure, pools):
        pd1 = keras.layers.Conv3D(
                filters=filters,
                kernel_size=kernel,
                strides=strides,
                padding="same"
                )
        pd2 = keras.layers.MaxPooling3D(
            pool_size=pool,
            padding="valid",
            )
        lys.extend((pd1,pd2))
        pd = pd2(pd1(pd))
    
    r1 = keras.layers.Reshape((prod(pd.shape[1:-1]) , pd.shape[-1]))(pd)
    
    dl1l = keras.layers.Dense(6, activation="sigmoid")(r1)
    dl1r = keras.layers.Dense(3, activation="leaky_relu")(r1)
    dl1 = keras.layers.Concatenate()([dl1r, dl1l])
    d1 = keras.layers.Permute((2,1))(dl1)
    
    dl2 = keras.layers.Dense(6*const.num_predict_per, activation="leaky_relu")(d1)
    dl3 = keras.layers.Dense(4*6*const.num_predict_per, activation="leaky_relu")(dl2)
    dl4 = keras.layers.MaxPooling1D(
        pool_size=2,
        data_format="channels_first",
        padding="valid",
        )(dl3)
    dl5 = keras.layers.Dense(2*6*const.num_predict_per, activation="leaky_relu")(dl4)
    dl6 = keras.layers.MaxPooling1D(
        pool_size=2,
        data_format="channels_first",
        padding="valid",
        )(dl5)
    
    o = keras.layers.Permute((2,1))(dl6)
    model = keras.Model(inputs=i, outputs=o)
    
    model.compile(
            optimizer=keras.optimizers.LossScaleOptimizer(keras.optimizers.Adamax(
                learning_rate=keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=3e-3,
                    decay_steps=1e3,
                    decay_rate=0.96,
                    staircase=False
                    )
                )),
            loss="mse",
            metrics=[
                "mae",
                #"mse",
                #"binary_crossentropy"
                ]
            )
    model.summary()
    return model

