import const
import keras
import read
import numpy as np
import model

observing = 0

xi, yi = read.data()
yi = yi[:, observing]
print(xi.shape)
print(yi.shape)
mdl = model.main()
mdl.fit(xi, yi, epochs=100, batch_size=5)
mdl.save(f"saved-models/{const.mdnum}.keras")

xtest, ytest = read.data(True)
ytest = ytest[:, observing]
mdl.evaluate(xtest, ytest)
