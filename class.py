import tensorflow as tf
from tf.keras.models import Model
class CharRNN(Model):
    def __init__(self, vocab_size, embedding_dim=64, lstm_units=128):
        super(CharRNN, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(lstm_units, return_sequences=False)
        self.dropout = Dropout(0.2)
        self.dense = Dense(vocab_size, activation='softmax')

    def call(self, x):
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.dropout(x)
        return self.dense(x)