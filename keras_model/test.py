import const
import read
import model
import keras
import sys

if len(sys.argv)>1:
    num = sys.argv[1]
else:
    num = const.mdnum

mdl = keras.models.load_model(f"./saved-models/{num}.keras")
mdl.summary()

head = 3

xtest, ytest = read.data(True)
mdl.evaluate(xtest, ytest)
print(mdl.predict(xtest)[0, :head])
print(ytest[0, :head])
