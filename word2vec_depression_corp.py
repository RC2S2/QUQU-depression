import os
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pandas as pd

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


TEXTDIR = None
sentences = MySentences(TEXTDIR)

model = gensim.models.Word2Vec(sentences, size=100, window=10, min_count=5, workers=4)
model.save("word2vec.model")

word_vectors = {key: model.wv[key] for key in model.wv.vocab}
vec = pd.DataFrame(word_vectors)
vec.to_csv("vec.csv", index=False)
