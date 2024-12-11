import const
import keras
import read
import numpy as np
import model

mdl = model.main()
xi, yi = read.data()
mdl.fit(xi, yi, epochs=100, batch_size=10)

xtest, ytest = read.data(True)
mdl.evaluate(xtest, ytest)
mdl.save(f"saved-models/{const.mdnum}.keras")
