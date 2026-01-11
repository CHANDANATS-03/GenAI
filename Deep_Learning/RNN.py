import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.layers import Input

#config
num_words=10000 #use tpo 10k words
maxlen=250#maxsequence length (pad/truncate)
embedding_dim=128
last_units=128
batch_size=64 # how many samples to process before updating model weights 
epochs=6
model_path="imdb_lstm.h5"

#lode dataset
(x_train,y_train),(x_test,y_test)=keras.datasets.imdb.load_data(num_words=num_words)
print("train samples:",len(x_train), "Test sample: ", len(x_test))

x_train=keras.preprocessing.sequence.pad_sequences(x_train,maxlen=maclen)
x_test=keras.preprocessing.sequence.pad_sequences(x_test,maxlen=maclen)


#build model
def build_model():
  inputs=keras.Input(shape=(maxlen,),dtype="int32")
  x=layers.Embedding(input_dim=num_words,output_dim=embedding_dim,input_length=maxlen)(inputs)
  x=layers.SpatialDropout1D(0.2)(x)
  x=layers.LSTM(last_units,return_sequences=False)(x)
  x=layers.Dropout(0.3)(x)
  outputs=layers.Dense(1, activation="sigmoid")(x)
  model=keras.Model(inputs,outputs,name="imdb_lstm")
  return model
model=build_model()
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"]
)
callback=[
    keras.callbacks.ModelCheckpoint(model_path,save_best_only=True,monitor="val_accuracy"),
    keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=2, restore_best_weights=True)
]

#val_loss --> validation loss. It is the loss calculated on the validation dataset during training
#val_accuracy--> validation accuracy. it is the accuracy calculated on the validation dataset during training
#patience=2 --> means that if the validation loss does not improve for 2 consecutive epochs, the training will be stopped 

#train
history=model.fit(
    x_train,y_train,
    validation_split=0.2,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callback
)
