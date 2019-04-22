#! /usr/bin/env python3

import os
import tensorflow as tf
import pandas as pd
import numpy as np

df = pd.read_csv("ml_data.csv")

training_dataset = (
    tf.data.Dataset.from_tensor_slices (
        tf.cast(df)
    )
)
