import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
import pickle
import nltk
from nltk.tokenize import sent_tokenize
#Download NLTK data(only first time)
nltk.download('punkt')
nltk.download('punkt_tab')
#1.Train model
corpus="""I love machine learning.I love deep learning.machine learning is fun.deep learning is powerful.I enjoy learning new things."""
#sentence tokenization using NLTK
sentences=sent_tokenize(corpus)
#Tokenize words
tokenizer=Tokenizer()
tokenizer.fit_on_texts(sentences)#fit on text will create word index
total_words=len(tokenizer.word_index)+1   #+1 for padding token
#prepare input sequences
input_sequences=[]
print('Sentences:',sentences)
for line in sentences:
  print("Preprocessing Lne:",line)
  token_list=tokenizer.texts_to_sequences([line])[0]  #convert text to sequence of tokens example
  print("Token List:",token_list)
  for i in range(1,len(token_list)):
    n_gram_sequence=token_list[:i+1] #create n-gram sequences example :[1,5],[1,5,23],[1,5,23,67]
    input_sequences.append(n_gram_sequence)
    print("N-gram Sequence:",n_gram_sequence)
    print("Current Input Sequence:",input_sequences)
print("Input Sequences:",input_sequences)

#pad sequences because they are of different lengths
max_sequence_len=max([len(x) for x in input_sequences])#Find the maximum lengths of sequences
input_sequences=pad_sequences(input_sequences,max_sequence_len,padding='pre')#pad sequece
print("Padded Input Sequences:",input_sequences)

#split into x and y
x=input_sequences[:, :-1]
print('x:',x)
y=tf.keras.utils.to_categorical(input_sequences[:,-1],num_classes=total_words)#convert to one-hot-encoding
print('y:',y)
#Build RNN model
model=Sequential([
    Embedding(total_words,10,input_length=max_sequence_len-1),  #embedding layer contains 10 dimensional
    SimpleRNN(100),#100 units in the RNN layer
    Dense(total_words,activation='softmax')
])
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])#compile the model
#Train
model.fit(x,y,epochs=200,verbose=1)#epoch means how many times the model will see the data
#save model & tokenizer
model.save('next_word_model.h5')
with open('tokenizer.pkl','wb') as f:
  pickle.dump((tokenizer,max_sequence_len),f)
print("Model & tokenizer saved successfully!")

#2 Load model and tokenizer
loaded_model=load_model('next_word_model.h5')
with open('tokenizer.pkl','rb') as f:
  loaded_tokenizer,loaded_max_sequence_len=pickle.load(f)
#3 Predict Function
def predict_next_word(seed_text):
  token_list=loaded_tokenizer.texts_to_sequences([seed_text])[0]  # convert seed text to sequence
  token_list=pad_sequences([token_list],maxlen=loaded_max_sequence_len-1,padding='pre')
  predicted=np.argmax(loaded_model.predict(token_list),axis=-1)  #Get the index of the word
  for word, index in loaded_tokenizer.word_index.items():  #Find the word corresponding to the previous
      if index==predicted:
        return f"{seed_text}->{word}"
  return "No prediction found"
#4 Test
print(predict_next_word("I love"))
print(predict_next_word("deep"))
print(predict_next_word("machine learning"))