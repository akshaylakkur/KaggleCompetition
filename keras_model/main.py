import const
import keras
import read
import numpy as np
import model

mdl = model.main()
xi, yi = read.data()
print(xi.shape)
print(yi.shape)
mdl.fit(xi, yi, epochs=500, batch_size=10)
mdl.save(f"saved-models/{const.mdnum}.keras")

xtest, ytest = read.data(True)
mdl.evaluate(xtest, ytest)
