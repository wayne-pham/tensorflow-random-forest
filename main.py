from pandas.core.frame import DataFrame
import tensorflow_decision_forests as tfdf
import pandas as pd

# set dataset
trainDataset = pd.read_csv("AmazonDataset3.csv")

# convert to tensorflow datasets
trainTFDS = tfdf.keras.pd_dataframe_to_tf_dataset(trainDataset, label="Helpfulness")

# Train Model
model = tfdf.keras.RandomForestModel()
model.fit(trainTFDS)

print(model.summary())