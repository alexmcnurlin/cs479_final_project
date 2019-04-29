#! /usr/bin/env python3
# https://www.tensorflow.org/tutorials/keras/basic_regression
import os
from sys import stdout
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import layers

tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

# Read dataset from CSV and convert all the relevant features to floats
df = pd.read_csv("ml_data.csv")
features = [
    'RequiredAge',
    'DemoCount',
    'DeveloperCount',
    'DLCCount',
    'MovieCount',
    'PackageCount',
    'RecommendationCount',
    'PublisherCount',
    'ScreenshotCount',
    'SteamSpyOwners',
    'SteamSpyOwnersVariance',
    'SteamSpyPlayersEstimate',
    'SteamSpyPlayersVariance',
    'AchievementCount',
    'AchievementHighlightedCount',
    'ControllerSupport',
    'IsFree',
    'FreeVerAvail',
    'PurchaseAvail',
    'SubscriptionAvail',
    'PlatformWindows',
    'PlatformLinux',
    'PlatformMac',
    'PCReqsHaveMin',
    'PCReqsHaveRec',
    'LinuxReqsHaveMin',
    'LinuxReqsHaveRec',
    'MacReqsHaveMin',
    'MacReqsHaveRec',
    'CategorySinglePlayer',
    'CategoryMultiplayer',
    'CategoryCoop',
    'CategoryMMO',
    'CategoryInAppPurchase',
    'CategoryIncludeSrcSDK',
    'CategoryIncludeLevelEditor',
    'CategoryVRSupport',
    'GenreIsNonGame',
    'GenreIsIndie',
    'GenreIsAction',
    'GenreIsAdventure',
    'GenreIsCasual',
    'GenreIsStrategy',
    'GenreIsRPG',
    'GenreIsSimulation',
    'GenreIsEarlyAccess',
    'GenreIsFreeToPlay',
    'GenreIsSports',
    'GenreIsRacing',
    'GenreIsMassivelyMultiplayer',
    'PriceInitial',
    'PriceFinal'
]
for feature in features:
    df[feature] = df[feature].astype(float)

# Setup training and test data
train_feature = "Metacritic"
df_test = df[df.Metacritic == 0][features]
df_train = df[df.Metacritic != 0][features + [train_feature]]
df_train_labels = df_train.pop("Metacritic")

# Get some statistics for later use
train_stats = df_train.describe()
train_stats = train_stats.transpose()
print(train_stats)

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(df_train)
normed_test_data = norm(df_test)

def build_model():
    model = tf.keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(df_train.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.keras.activations.sigmoid),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer,
        metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

model = build_model()
# model.summary()


# Display training progress by printing a single dot for each completed epoch
class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 25 == 0:
            print(f" Epoch {epoch}")
        print('.')

EPOCHS = 25
print("\n"*20)

history = model.fit(
    df_train,
    df_train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[PrintDot()]
)

df_batch = df_test
test_batch = model.predict(df_batch)
i = 0
for index, row in df_batch[:20].iterrows():
    print(df.loc[index, "QueryName"], " => ", test_batch[i])
    i += 1

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

results = model.predict(df_test)

def plot_history(history):
    hist = pd.DataFrame(history.history)
    print("Final mean_absolute_error",hist["mean_absolute_error"].tail())
    hist['epoch'] = history.epoch

    plt.figure()
    plt.hist(results, bins=range(0, 101, 1))
    plt.title("Distribution of predicted ratings")
    plt.xlabel('Metacritic Rating')
    plt.ylabel('Number of Entries')

    plt.figure()
    plt.title("Average Error in Prediction")
    plt.xlabel('Epoch')
    plt.yscale("log")
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
        label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
        label = 'Val Error')
    plt.legend()

    plt.figure()
    plt.title("Average Error Squared in Prediction")
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.yscale("log")
    plt.plot(hist['epoch'], hist['mean_squared_error'],
        label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
        label = 'Val Error')
    plt.legend()
    plt.show()


plot_history(history)
IPython.embed()

