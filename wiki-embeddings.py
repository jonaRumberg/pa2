from datasets import load_dataset

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

#Load at max 1000 documents + embeddings
max_docs = 3516800
docs_stream = load_dataset(f"Cohere/wikipedia-22-12-en-embeddings", split="train", streaming=True)

docs = []
doc_embeddings = []
i = 0

for doc in docs_stream:
    if (doc["paragraph_id"] == 0):
        docs.append(doc)
        doc_embeddings.append(doc['emb'])
    print(f"Indexed {round(i/max_docs * 1000)/10} % of documents. Loaded {len(docs)}", end="\r")

    i += 1
    if i >= max_docs:
        break

print(f"Loaded {len(docs)} documents")

docs = pd.DataFrame(docs)
print("asdfasdf")

X = np.array(docs["emb"].to_list(), dtype=np.float32)

tsne = TSNE(random_state=0, n_iter=1000)
tsne_results = tsne.fit_transform(X)

df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
df_tsne['Text'] = docs['text'].tolist()
df_tsne['Title'] = docs['title'].tolist()
df_tsne['wiki_id'] = docs['wiki_id'].tolist()
print(df_tsne)

px.scatter(
    data_frame=df_tsne, 
    x='TSNE1', 
    y='TSNE2', 
    hover_data=['Title', 'wiki_id'], 
    title='Scatter plot of embeddings using t-SNE'
    ).show()
