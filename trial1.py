#! /usr/bin/env python


import tensorflow as tf
import tensorflow_hub as hub
import os
import numpy as np

print('start')
os.environ["TFHUB_CACHE_DIR"] = r'../cache'


embed = hub.load(r"../cache/f4ea2eb4a9fd72946209ef45271146fae070fb29")

embeddings = embed([
    "The quick brown fox jumps over the lazy dog."])["outputs"]
arr = np.array(embeddings)

print(len(arr))
print(arr[0])

#print(np.array(embeddings[0]))

print('end')