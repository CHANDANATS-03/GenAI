''' Bidirectional RNN'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM,  Dense

corpus=[
    "i love deep learning",
    "i love natural language processing",
    "i enjoy learning new things",
    "deep learning models are powerful"
    "language models can predict words"
]
#Tokenize text
tokenizer=Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words=len(tokenizer.word_index)+1
print("vocabulary size", total_words)
input_sequences=[]
for line in corpus:
  token_list = tokenizer.texts_to_sequences([line])[0]
  for i in range(1, len(token_list)):
    n_gram_seq = token_list[:i+1]
    input_sequences.append(n_gram_seq)

#pad sequence to equal length
max_seq_len= max([len(x) for x in input_sequences])
input_sequences=np.array(pad_sequences(input_sequences,maxlen=max_seq_len,padding='pre'))

X= input_sequences[:, :-1]#features
y= input_sequences[:, -1] #labels
y=tf.keras.utils.to_categorical(y, num_classes=total_words)
print(X.shape)
print(y.shape)

model=Sequential()
model.add(Embedding(total_words,64,input_length=max_seq_len-1))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(total_words,activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])

def predict_next_word(seed_text,next_words=1):
  for _ in range(next_words):
    token_list= tokenizer.texts_to_sequences([seed_text])[0]
    token_list=pad_sequences([token_list], maxlen=max_seq_len-1,padding='pre')
    predicted=np.argmax(model.predict(token_list), axis=-1)[0]

    for word, index in tokenizer.word_index.items():
      if index== predicted:
        seed_text += " "+ word
        break
    return seed_text
  
print(predict_next_word("i love",3))
print(predict_next_word("deep learning",3))