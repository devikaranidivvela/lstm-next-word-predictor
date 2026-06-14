# load the model and libraries
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


#load the pre-trained model
model=load_model('next_word_prediction_model.h5')

#load the tokenizer
with open('tokenizer.pkl','rb') as f:
  tokenizer=pickle.load(f)

#function to predict next word
#function to predict next word
def predict_next_word(seed_text,model,tokenizer,max_sequence_len):
    token_list=tokenizer.texts_to_sequences([seed_text])[0]
    if len(token_list)>=max_sequence_len:
        token_list=token_list[-(max_sequence_len-1):]
    token_list=pad_sequences([token_list],maxlen=max_sequence_len,padding='pre')
    predicted=np.argmax(model.predict(token_list),axis=1)
    for word,index in tokenizer.word_index.items():
        if index==predicted:
            return word
    return None


#streamlit app
import streamlit as st
st.title('Next word prediction with LSTM')
st.write('Enter a sequence of words:')

input_text=st.text_area('Enter your review here')

if st.button('Predict'):
  max_sequence_len=model.input_shape[1]+1
  next_word=predict_next_word(input_text,model,tokenizer,max_sequence_len)
  st.write(f'Next word: {next_word}')
