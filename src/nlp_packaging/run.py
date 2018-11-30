import spacy

import matplotlib.pyplot as plt
import seaborn as sns
from processing import tf_idf_scores
from data import texts

nlp = spacy.load("en")

docs = [nlp(text) for text in texts]

res = tf_idf_scores(docs)

sns.set()

fig, ax = plt.subplots(figsize=(15,3))
sns.heatmap(res, ax=ax)
plt.show()