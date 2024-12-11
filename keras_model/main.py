import const
import keras
import read
import numpy as np
import model

mdl = model.main()
xi, yi = read.data()
mdl.fit(xi, yi, epochs=1000, batch_size=10)

xtest, ytest = read.data(True)
#print(xtest.shape)
#print(ytest.shape)
mdl.evaluate(xtest, ytest)
#print(mdl.predict(xtest))
#print(ytest)
mdl.save(f"saved-models/{const.mdnum}.keras")
