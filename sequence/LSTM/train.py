import collections
import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed, LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np


def read_words(filename):
    with open(filename, "r") as f:
        text = f.read()
        text = text.replace("\n", " ")
        text = text.replace("<unk>", " ")
        return [w for w in text.split() if len(w) > 3]


def build_vocab(filename):
    data = read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def load_data():
    word_to_id = build_vocab("ptb.train.txt")
    train_data = file_to_word_ids("ptb.train.txt", word_to_id)
    valid_data = file_to_word_ids("ptb.valid.txt", word_to_id)
    vocabulary = len(word_to_id)
    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
    return train_data, valid_data, vocabulary, id_to_word

train_data, valid_data, vocabulary, id_to_word = load_data()

class KerasBatchGenerator(object):
    def __init__(self, data, num_steps, batch_size, vocabulary):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.current_idx = 0

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                y[i, :] = to_categorical(self.data[self.current_idx + self.num_steps], num_classes=self.vocabulary)
                self.current_idx += self.num_steps
            yield x, y

num_steps = 20
batch_size = 20
train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary)
valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size, vocabulary)


hidden_size = 500
model = Sequential()
model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(vocabulary))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
print(model.summary())

num_epochs = 10
model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
                    validation_data=valid_data_generator.generate(),
                    validation_steps=len(valid_data)//(batch_size*num_steps))

model.save("test.h5")