
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

gravedad=np.array([1.014 ,1 , 0.934 , 0.876 , 0.825 , 0.780], dtype=float)

API=np.array([8,10,20,30,40,50],dtype=float)

# capa=tf.keras.layers.Dense(units=1, input_shape=[1])
# modelo=tf.keras.models.Sequential([capa])


model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64,input_shape=[1],activation="relu"))
model.add(tf.keras.layers.Dense(32,activation="relu"))
model.add(tf.keras.layers.Dense(1))

model.compile (
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss="mse"

)

# print("Comenzando el entrenamiento...")

historial=model.fit(API,gravedad,epochs=300,verbose=True)



# print("Prediccion")

# predicho=modelo.predict([18])
# print(predicho[0][0])



# plt.plot(historial.history["loss"])
# plt.plot(historial.history["loss"])
# plt.show()

# model.save("model.h5")

# from keras.models import load_model

# modelo=load_model("model.h5")

# resultado=modelo.predict([10])

# print("Resultado")
# print(resultado[0][0])