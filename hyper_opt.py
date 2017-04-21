from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#%matplotlib inline

#for plots
def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(size=6)
    ax.yaxis.set_tick_params(size=6)
colors=['#F5A21E', '#02A68E', '#EF3E34', '#134B64', '#FF07CD']
#########

num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

##########

def build_model(opt=keras.optimizers.Adadelta()):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    return model
############
# First define a callback to record train and test loss on each minibatch
from keras.callbacks import Callback


class minibatch_History(Callback):
    """Callback that records events into a `History` object.

    Predicts over the validation set and each input batch (w/o dropout)
    after each batch.

    """
    def __init__(self, count_mode='samples', Nevery = 1):
        super(minibatch_History, self).__init__()
        self.Nevery = Nevery


    def on_train_begin(self, logs=None):
        self.batch = []
        self.history = {'val_loss':list(),
                        'val_acc':list(),
                        'train_loss':list(),
                        'train_acc':list()}
        self.batch_no = 0
        self.target = self.params['samples']

    def on_batch_end(self, batch, logs=None):
        if self.batch_no % self.Nevery == 0 or self.batch_no == self.target:
            logs = logs or {}
            self.batch.append(batch)
            #print([np.array(l).shape for l in logs['input_batch']])
            for k, v in logs.items():
                if k is not 'input_batch':
                    self.history.setdefault(k, []).append(v)

            # add validation loss. Only test on a random subset of minibatch size
            val_loss, val_acc =  self.model.evaluate(self.validation_data[0],
                                                     self.validation_data[1], verbose=0)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # add train loss
            train_loss, train_acc =  self.model.evaluate(logs['input_batch'][0],
                                                     logs['input_batch'][1], verbose=0)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
###########
from hyperopt import fmin, hp, Trials, tpe, STATUS_OK

space4rf = {
    'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-1)),
    'var_care': hp.loguniform('var_care', np.log(1e-1),np.log(1e1)),
    'momentum': hp.choice('momentum', [0,1]),
    'sqrt': hp.choice('sqrt', [0,1]),
    'pn': hp.choice('pn', [-1.,1.]),
}

grava_trials = Trials()

def fnc(params):
    gr_model = Sequential()
    gr_model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    gr_model.add(Conv2D(64, (3, 3), activation='relu'))
    gr_model.add(MaxPooling2D(pool_size=(2, 2)))
    gr_model.add(Dropout(0.25))
    gr_model.add(Flatten())
    gr_model.add(Dense(128, activation='relu'))
    gr_model.add(Dropout(0.5))
    gr_model.add(Dense(num_classes, activation='softmax'))


    opt = keras.optimizers.GraVa(**params)
    gr_model.compile(loss=keras.losses.categorical_crossentropy,
                                   optimizer=opt,
                                      metrics=['accuracy'])

    gr_model.fit(x_train[:10000], y_train[:10000],
          batch_size=32,
          epochs=10,
          verbose=0,

          validation_data=(x_test[:1000], y_test[:1000]))
    score = gr_model.evaluate(x_test, y_test, verbose=0)
    print(score, params)
    return {'loss':score[0], 'status': STATUS_OK }

hyperoptBest = fmin(fnc, space4rf, algo=tpe.suggest, max_evals=200, trials=grava_trials)

grava_trials_1 = Trials()

def fnc(params):
    gr_model = Sequential()
    gr_model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    gr_model.add(Conv2D(64, (3, 3), activation='relu'))
    gr_model.add(MaxPooling2D(pool_size=(2, 2)))
    gr_model.add(Dropout(0.25))
    gr_model.add(Flatten())
    gr_model.add(Dense(128, activation='relu'))
    gr_model.add(Dropout(0.5))
    gr_model.add(Dense(num_classes, activation='softmax'))


    opt = keras.optimizers.GraVa(**params)
    gr_model.compile(loss=keras.losses.categorical_crossentropy,
                                   optimizer=opt,
                                      metrics=['accuracy'])

    gr_model.fit(x_train[:10000], y_train[:10000],
          batch_size=1,
          epochs=10,
          verbose=0,

          validation_data=(x_test[:1000], y_test[:1000]))
    score = gr_model.evaluate(x_test, y_test, verbose=0)
    print(score, params)
    return {'loss':score[0], 'status': STATUS_OK }

hyperoptBest = fmin(fnc, space4rf, algo=tpe.suggest, max_evals=200, trials=grava_trials_1)


space4rf = {
    'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-1)),
    'hess': hp.loguniform('hess', np.log(1e-3),np.log(1e0)),
    'm1': hp.choice('m1', [0,.9]),
    'm2': hp.choice('m2', [0,.9]),
}

coord_trials = Trials()

def fnc(params):
    gr_model = Sequential()
    gr_model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    gr_model.add(Conv2D(64, (3, 3), activation='relu'))
    gr_model.add(MaxPooling2D(pool_size=(2, 2)))
    gr_model.add(Dropout(0.25))
    gr_model.add(Flatten())
    gr_model.add(Dense(128, activation='relu'))
    gr_model.add(Dropout(0.5))
    gr_model.add(Dense(num_classes, activation='softmax'))


    opt = keras.optimizers.CoordDescent(**params)
    gr_model.compile(loss=keras.losses.categorical_crossentropy,
                                   optimizer=opt,
                                      metrics=['accuracy'])

    gr_model.fit(x_train[:10000], y_train[:10000],
          batch_size=64,
          epochs=10,
          verbose=0,

          validation_data=(x_test[:1000], y_test[:1000]))
    score = gr_model.evaluate(x_test, y_test, verbose=0)
    print(score, params)
    return {'loss':score[0], 'status': STATUS_OK }

#hyperoptBest = fmin(fnc, space4rf, algo=tpe.suggest, max_evals=200, trials=coord_trials)




##########
import pickle

pickle.dump( [grava_trials, grava_trials_1, coord_trials], open( "grava_opt.p", "wb" ) )
