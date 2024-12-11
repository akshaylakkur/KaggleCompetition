import const
import keras
import numpy as np

#@keras.saving.register_keras_serializable("compare")
#def compare(yt, yp):
#    print(y_true.shape)
#    print(y_pred.shape)
#    
#    ypd = []
#    for plane in y_pred[0]:
#        for line in plane:
#            for pt in line:
#                for i,prot in enumerate(pt):
#                    ypd.append([*prot[1:]]+[0]*i+[prot[0]]+[1]*(5-i))
#    
#    def dist(a, b):
#        return ((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)**0.5
#    def dot(a, b):
#        out = 0
#        for av, bv in zip(a,b):
#            out += av*bv
#        return out
##    def itr():
##        for plane in y_pred:
##            for line in plane:
##                for pt in line:
##                    yield pt[0],(pt[1], pt[2], pt[3])
#    
#    exp = -3
#    penalty = 0
#
#    answs = np.zeros((len(ypd),len(yt)), dtype=np.float32)
#    for i,(xp,yp,zp,*whichp) in enumerate(ypd):
#        small_sm = 0
#        small_ct = 0
#        for j,(xt,yt,zt,*whicht) in enumerate(yt):
#            act = dot(whichp, whicht)
#            add = (act*dist([xt,yt,zt],[xp,yp,zp]))**exp
#            answs[i,j] = add
#    penalty = 0
#    for row in answs:
#        penalty += 3*(row.mean())**(1/exp)
#    for col in answs.transpose():
#        penalty += 10*(col.mean())**(1/exp)
##    for xp,yp,zp,*whichp in yp:
##        for xt,yt,zt,*whicht in yt
##        
##        sm = 0
##        num = 0
##        for j in yt:
##            sm += (dist(i,j)*act)**exp
##            num += 1
##        penalty += 3*(sm/num)**(1/exp)
##    
##    for x,y,z,*which in yt:
##        
##        sm = 0
##        num = 0
##        for act,j in yp:
##            sm += (dist(i,j)*act)**exp
##        penalty += 10*(sm/num)**(1/exp)
#    return penalty

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
    
    r2 = keras.layers.Permute((2,1))(dl1)
    
    #d1 = keras.layers.Dense(_num_predict//10)(r)
    dl2 = keras.layers.Dense(const.num_predict, activation="leaky_relu")(r2)
    #d1 = keras.layers.Conv1D(
    #        filters=const.num_predict,
    #        kernel_size=10,
    #        padding="same",
    #        activation="sigmoid")(r)
    
    d = dl2
    o = keras.layers.Permute((2,1))(d)
    model = keras.Model(inputs=i, outputs=o)
    
    '''
    # Back pass: find starting weights
    dcv = pd
    weights = [dcv]
    for (filters, kernel, strides), pool in zip(
            structure[-2::-1],
            pools[::-1]):
        dc1 = keras.layers.UpSampling3D(pool)(dcv)
        dcv = keras.layers.Conv3DTranspose(
                filters=filters,
                kernel_size=kernel,
                strides=strides,
                padding="same"
                )(dc1)
    model = keras.Model(inputs=i1, outputs=dcv)
    
    # Average calculation
    b = np.arange(const.batch)
    xs = np.arange(dcv.shape[2])
    ys = np.arange(dcv.shape[3])
    zs = np.arange(dcv.shape[1])
    ch = np.array([1])
    bb, zz, xx, yy, cc = np.meshgrid(zs, b, xs, ys, ch)
    xw = keras.layers.Multiply()([dcv, xx])
    yw = keras.layers.Multiply()([dcv, yy])
    zw = keras.layers.Multiply()([dcv, zz])
    for l in lys[2:]:
        xw = l(xw)
        yw = l(yw)
        zw = l(zw)
    pd = keras.layers.Reshape(pd.shape[1:]+(1,))(pd)
    xw = keras.layers.Reshape(xw.shape[1:]+(1,))(xw)
    yw = keras.layers.Reshape(yw.shape[1:]+(1,))(yw)
    zw = keras.layers.Reshape(zw.shape[1:]+(1,))(zw)
    avgs = keras.layers.Concatenate()([pd, xw, yw, zw])
    model = keras.Model(inputs=i1, outputs=avgs)
    '''
    
    model.compile(
            optimizer=keras.optimizers.LossScaleOptimizer(keras.optimizers.Nadam(
                learning_rate=keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=3e-3,
                    decay_steps=1e4,
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

