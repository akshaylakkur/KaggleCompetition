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

xtest, ytest = read.data(True)
mdl.evaluate(xtest, ytest)

head = 3
tail = 3
pred = mdl.predict(xtest)[0, 0]
tru = ytest[0, 0]
print(pred[:head])
print(tru[:head])
print(pred[-tail:])
print(tru[-tail:])
