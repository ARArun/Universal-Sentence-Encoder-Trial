#! /usr/bin/env python

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
from sklearn.metrics.pairwise import cosine_similarity

print('start')
os.environ["TFHUB_CACHE_DIR"] = r'../cache'
embed = hub.load(r"../cache/f4ea2eb4a9fd72946209ef45271146fae070fb29")


repo = pd.read_json(r'../Data/embedded.json')
repo['TitleEmbedding'] = repo['TitleEmbedding'].apply(lambda x: np.asarray(x).reshape(1,512))
print(repo.head())

query = input('Enter Your Query Please?\t')

query = [query]
queryEmbedding = np.array(embed(query)['outputs'])[0]
queryEmbedding = queryEmbedding.reshape(1,512)
print(queryEmbedding.shape)



repo['CosSim'] = repo['TitleEmbedding'].apply(lambda x: cosine_similarity(queryEmbedding, x))

print(repo.head())