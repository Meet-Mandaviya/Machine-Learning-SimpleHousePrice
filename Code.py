import tensorflow as tf #it is a machine learning library
import numpy as np
from tensorflow import keras # keras is a library used for neural networks

model = tf.keras.Sequential([keras.layers.Dense(units= 1, input_shape =[1])])
model.compile(optimizer='sgd',loss = 'mean_squared_error')

xs = np.array([1.0,2.0,3.0,4.0,5.0,6.0],dtype = float)
ys = np.array([1.0,1.5,2.0,2.5,3.0,3.5],dtype = float)

model.fit(xs,ys,epochs=1000)
print(model.predict([7.0]))
