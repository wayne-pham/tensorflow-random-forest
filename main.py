from pandas.core.frame import DataFrame
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import tensorflow as tf
import math

# set dataset
trainDataset = pd.read_csv("AmazonDatasetModified.csv")

# convert to tensorflow datasets
trainTFDS = tfdf.keras.pd_dataframe_to_tf_dataset(trainDataset, label="ReviewTextWordCount", max_num_classes=200)

# Train Model
model = tfdf.keras.RandomForestModel()
# model.compile(metrics=["accuracy"])

model.fit(x=trainTFDS)

model.make_inspector().evaluation()

print(model.summary())

tfdf.model_plotter.plot_model(model)
