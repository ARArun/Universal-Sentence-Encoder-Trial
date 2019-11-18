#! ../ten/bin/python3

import tensorflow as tf
import tensorflow_hub as hub
import os
import pandas as pd
import numpy as np

print('start')
os.environ["TFHUB_CACHE_DIR"] = r'../cache'


embed = hub.load(r"../cache/f4ea2eb4a9fd72946209ef45271146fae070fb29")

def embedding(x):
    x = [x]
    EmbeddingList = embed(x)['outputs']
    return np.array(EmbeddingList[0])


data = pd.read_csv(r'../Data/data.csv', index_col=0)
print(data.head())

data['TitleEmbedding'] = data['title'].apply(lambda x: embedding(x))

print(data.head())

data.to_json(r'../Data/embedded.json')

print('end')