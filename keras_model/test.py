import const
import read
import model
import keras

mdl = keras.models.load_model(f"saved-models/{const.mdnum}.keras")
mdl.summary()

head = 3

xtest, ytest = read.data(True)
mdl.evaluate(xtest, ytest)
print(mdl.predict(xtest)[0, :head])
print(ytest[0, :head])
