# This file is about to build a RNN(GUR, in perticular) model to 
# learn a bloom filter which copy the instruction showned in 
# THE CASE FOR LEARNED INDEX MODEL.

import numpy as np
from collections import Counter
from AbstractModel import AbstractModel
import keras


class GRUModel(AbstractModel):

    def __init__(self, embedding_path=None,
                    embedding_dim=None,
                    maxlen=50,
                    lr=0.001, 
                    batch_size=256, gru_size=16, 
                    decay=0.0001, epochs=30, lstm=False, 
                    dense_only=False, second_gru_size=None):
        self.embedding_path = embedding_path
        self.embedding_dim = embedding_dim
        self.maxlen = maxlen
        self.lr = lr
        self.batch_size = batch_size
        self.gru_size = 16
        self.decay = 0.0001
        self.epochs = 30
        self.lstm = False
        self.dense_only = False
        self.second_gru_size = second_gru_size


    def fit(self, text_X, status_y):
        # process the input data to model usable vector
        X, y = self.vectorize_dataset(text_X, status_y)
        num_chars = len(self.char_indices)

        # pre_layers only contains the setting of embedding layers
        pre_layers = []
        
        if embedding_path:
            embedding_vectors = {}
            
            with open(self.embedding_path, 'r') as f:
                for line in f:
                    line_split = line.strip().split(" ")
                    vec = np.array(line_split[1:], dtype=float)
                    char = line_split[0]
                    embedding_vectors[char] = vec

            # the embedding matrix left index 0 as "mask_zero"
            embedding_matrix = np.zeros((num_chars+1, self.embedding_dim))

            for char, i in self.char_indices.items():
                vec = embedding_vectors.get(char)
                assert(vec is not None)
                embedding_matrix[i] = vec

            # the embedding layer will be update during the training process
            # For how embedding vector are learned visit 
            # https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526
            print("the embedding matrix has shape: ", embedding_matrix.shape)
            pre_layers.append(keras.layers.Embedding(
                num_chars+1, self.embedding_dim, input_length=self.maxlen, weights=[embedding_matrix]
            ))

        # setting for the NN
        post_layers = []

        # processing the neural network
        if self.lstm:
            rnn_layer = keras.layers.LSTM(self.gru_size,
                                          return_sequences=False if not self.second_gru_size else True)
        else:
            rnn_layer = keras.layers.GRU(self.gru_size,
                                         return_sequences=False if not self.second_gru_size else True)

        post_layers.append(rnn_layer)


        if self.second_gru_size:
            post_layers.append(keras.layers.GRU(self.second_gru_size))

        if self.hidden_size:
            post_layers.append(keras.layers.Dense(self.hidden_size, activation='relu'))

        post_layers += [
            keras.layers.Dense(1),
            keras.layers.Activation('sigmoid'),
        ]

        if not self.dense_only:
            layers = pre_layers + post_layers
        else:
            # only use the embedding layer and FCN if dense only set to true
            layers = pre_layers + [
                keras.layers.Flatten(),
                keras.layers.Dense(8, activation='relu'),
                keras.layers.Dense(4, activation='relu'),
                keras.layers.Dense(2, activation='relu'),
                keras.layers.Dense(1, activation='sigmoid'),
            ]

        self.model = keras.models.Sequential(layers)
        # using rmsprop as it is usually a good choice for RNN
        optimizer = keras.optimizers.rmsprop(lr=self.lr, decay=self.decay)

        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, verbose=2)


    def predict(self, items):
        """
        Predict the existence of list of URLS
        """

        vectorized_data = np.zeros((len(items), self.maxlen), dtype=np.int)

        # use pre-padding and pre-truncated as Keras default setting
        for i in range(len(items)):
            offset = max(self.maxlen - len(items[i]), 0)
            for t, char in enumerate(items[i]):
                if t >= self.maxlen:
                    break
                vectorized_data[0, t+offset] = self.char_indices[char]

        return [pred[0] for pred in self.model.predict(vectorized_data)]


    def vectorize_dataset(self, text_X, status_y):
        """
        Vectorized String (more precisely for our case, URL data) 
        which could be used to train a character-level RNN.

        text_X :
            A list of all the URLs
        
        status_y:
            A bitarray that represent the existence of each URL in text_X

        maxlen:
            We only record the first n appeared char in each URL.
        """
        
        print("Concating Urls...")
        raw_text = ''.join(text_X)
        chars = sorted(list(set(raw_text)))
        print('Total chars used in URL set:', len(chars))

        self.char_indices = dict((c, i + 1) for i, c in enumerate(chars))

        # 0 in this indicates empty word, 1 through len(chars) inclusive
        # indicates a particular char
        X = np.zeros((len(text_X), self.maxlen), dtype=np.int)
        y = np.zeros((len(text_X)), dtype=np.bool)

        # use pre-padding and pre-truncated as Keras default setting
        for i, url in enumerate(text_X):
            offset = max(self.maxlen - len(url), 0)
            for t, char in enumerate(url):
                if t >= self.maxlen:
                    break
                X[i, t + offset] = self.char_indices[char]
            y[i] = 1 if status_y[i] == 1 else 0

        return X, y

if __name__ == "__main__":
    pass
