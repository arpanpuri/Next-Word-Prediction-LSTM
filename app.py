import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the LSTM Model
model = load_model('next_word_lstm.h5')

#3 Laod the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# streamlit app
st.set_page_config(page_title="Next Word Prediction", layout="centered")
st.title("Next Word Prediction With LSTM And Early Stopping")

st.markdown("""
Welcome! This app uses a trained LSTM model to predict the next word in a given sequence. Enter a partial sentence and click **Predict Next Word** to see what comes next.

**Examples to try:**
- The quick brown fox
- Once upon a time
- Deep learning is
- To be or not to
""")

input_text = st.text_input("Enter the sequence of words", "To be or not to")

if st.button("Predict Next Word"):
    if not input_text.strip():
        st.warning("Please enter a sequence of words.")
    else:
        max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        if next_word:
            st.success(f"**Next word:**  ")
            st.markdown(f"<div style='font-size:2em; color:#4CAF50; font-weight:bold'>{next_word}</div>", unsafe_allow_html=True)
        else:
            st.error("Sorry, could not predict the next word. Try a different input.")

