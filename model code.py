import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import requests

# 1. Load and preprocess the dataset
def load_dataset():
    url = "https://www.gutenberg.org/files/1661/1661-0.txt"
    response = requests.get(url)
    text = response.text
    return text

def preprocess_dataset(text):
    # Preprocess the dataset (tokenization, remove punctuation, etc.)
    text = text.lower()
    tokenizer = Tokenizer(char_level=False, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts([text])
    return tokenizer, tokenizer.texts_to_sequences([text])[0]

def create_sequences(tokenized_text, sequence_length):
    input_sequences = []
    output_sequences = []

    for i in range(len(tokenized_text) - sequence_length):
        input_sequences.append(tokenized_text[i:i + sequence_length])
        output_sequences.append(tokenized_text[i + sequence_length])

    return np.array(input_sequences), np.array(output_sequences)

text = load_dataset()
tokenizer, tokenized_text = preprocess_dataset(text)
input_sequences, output_sequences = create_sequences(tokenized_text, sequence_length=100)

# 2. Build the RNN model
def build_rnn(vocab_size, embedding_dim, rnn_units, batch_size):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        Dense(vocab_size)
    ])
    return model

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 256
rnn_units = 1024
batch_size = 64

model = build_rnn(vocab_size, embedding_dim, rnn_units, batch_size)
model.summary()

# 3. Train the model
def train_model(model, input_sequences, output_sequences, batch_size, epochs):
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    history = model.fit(input_sequences, output_sequences, batch_size=batch_size, epochs=epochs)
    return history

history = train_model(model, input_sequences, output_sequences, batch_size, epochs=10)

# 4. Visualize the training progress
def visualize_training(history):
    plt.plot(history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

visualize_training(history)

# 5. Generate text using the trained model
def generate_text(model, start_string, num_generate, temperature):
    input_eval = [tokenizer.word_index[s] for s in start_string.split()]
    input_eval = tf.expand_dims(input_eval, 0)

    generated_text = []
    model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        generated_text.append(tokenizer.index_word[predicted_id])

    return ' '.join(generated_text)

generated_text = generate_text(model, start_string="Sherlock Holmes", num_generate=1000, temperature=1.0)
print(generated_text)