import const
import read
import keras

mdl = keras.models.load_model("saved-models/{const.mdnum}.keras")
mdl.summary()

x, y = read.data(True)
print(mdl.predict(x)[0, :5])
print(y[0, :5])
