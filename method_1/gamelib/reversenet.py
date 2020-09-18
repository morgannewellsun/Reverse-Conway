import os
import numpy as np
import pandas as pd
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from gamelib.logutil import output_root


class ReverseNet:
    
    def __init__(self, board_size, layer_spec, batches = 64, epochs = 200):
        self.density = 0.15
        self._board_size = board_size
        self.model = models.Sequential()
        self.model.add(layers.Dense(
            layer_spec[0],
            activation = 'relu',
            input_shape = (board_size+1,)
            ))
        for sz in layer_spec[1:]:
            self.model.add(layers.Dense(sz, activation = 'relu'))
            self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(board_size, activation = 'sigmoid'))
        self.model.compile(optimizer = 'rmsprop', 
                      loss = 'binary_crossentropy', 
                      metrics = ['accuracy'])
        self._batches = batches
        self._epochs = epochs
        self._model_files = output_root + 'nn_model'
        self._predict_dir = output_root + 'nn_predict/'

    
    def was_trained(self):
        return os.path.exists(self._model_files)


    def train(self, stop_states, start_states):
        """ stop_states is a list with first element being delta.
        """
        # epochs = int(stop_states.shape[0] / self._batches) * 2 + 2
        self._hist = self.model.fit(
            x = stop_states,
            y = start_states,
            batch_size = self._batches,
            epochs = self._epochs,
            verbose = 2,
            validation_split = 0.2)
        os.makedirs(self._model_files)
        self.model.save(self._model_files)
        self.print_summary(len(start_states))


    def load(self):
        self.model = models.load_model(self._model_files)


    def print_summary(self, data_size):
        with open(self._predict_dir + 'summary.txt','w+') as fh:
            self.model.summary(print_fn = lambda x: fh.write(x + '\n'))
            fh.write('\nUsed data size {}\n'.format(data_size))


    def display_train(self):
        loss = self._hist.history['loss']
        val_loss = self._hist.history['val_loss']
        eposet = range(1, len(loss)+1)
        plt.plot(eposet, loss, 'bo', label='Training loss')
        plt.plot(eposet, val_loss, 'b', label='Validation loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        # plt.show()
        plt.savefig(self._predict_dir + 'train_perform.pdf')


    def revert(self, stop_states, tofile = True):
        if not os.path.exists(self._predict_dir):
            os.makedirs(self._predict_dir)
        prob = self.model.predict(stop_states)
        df = pd.DataFrame(prob, index=stop_states.index)
        df.insert(0, 'delta', stop_states['delta'])
        if tofile:
            df.to_csv(self._predict_dir + 'prob.csv')
        for delta in set(stop_states.delta):
            strip = (df['delta']==delta)
            group_data = df.loc[strip, df.columns[1:]].to_numpy().flatten()
            rank = int(len(group_data) * self.density)
            ind = np.argpartition(group_data, -rank)[-rank:]
            threshold = min(group_data[ind])
            df.loc[strip, df.columns[1:]] = (df.loc[strip, df.columns[1:]] > threshold)
        df.iloc[:, 1:] = df.iloc[:, 1:].astype(int)
        if tofile:
            df.to_csv(self._predict_dir + 'predict.csv')
        return (prob, df)



