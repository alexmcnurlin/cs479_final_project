#! /usr/bin/env python3
# https://www.tensorflow.org/tutorials/keras/basic_regression

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
from tensorflow.keras import layers

tf.enable_eager_execution()

df = pd.read_csv("ml_data.csv")
features = [
    'IsFree',
    'FreeVerAvail',
    'PurchaseAvail',
    'SubscriptionAvail',
    'PlatformWindows',
    'PlatformLinux',
    'PlatformMac'
]
df_test = df[df.Metacritic == 0]
df_train = df[df.Metacritic != 0]

# print(df_test)
# print(df_train)

dataset_train = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(df_train[features].values, tf.float32),
            tf.cast(df_train["Metacritic"].values, tf.int32)
        )
    )
)

dataset_test = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(df_test[features].values, tf.float32),
            tf.cast(df_test["Metacritic"].values, tf.int32)
        )
    )
)


# TODO: Normalize data?
def norm(x):
    return (x - dataset_train['mean']) / dataset_train['std']
normed_train_data = norm(dataset_train)
normed_test_data = norm(dataset_test)

def build_model():
    model = tf.keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(df.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

model = build_model()
model.summary()


# Display training progress by printing a single dot for each completed epoch
class PrintDot(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0:
                print('')
            print('.', end='')

EPOCHS = 1000

#history = model.fit(
 #   dataset_train,
  #  train_labels,
   # epochs=EPOCHS,
    #validation_split=0.2,
    #verbose=0,
    #callbacks=[PrintDot()]
#)
