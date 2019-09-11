#if want to use spacy, download follwing
#python -m spacy download en

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, CuDNNGRU, CuDNNLSTM, Bidirectional, GRU, BatchNormalization
from keras.initializers import Constant
import warnings
warnings.filterwarnings(action='ignore')

df = pd.read_excel('data.xlsx', encoding='utf-8')
#df.sample(frac=1).reset_index(drop=True)
x_total = df['text'].values

max_length = max([len(s.split()) for s in x_total])

normalized_text = []
for string in x_total.tolist():
     tokens = re.sub("[^a-z0-9']+", " ", string.lower())
     normalized_text.append(tokens)

temp=[]
result=[]

for sentence in normalized_text:
    tokens = word_tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    temp = [w for w in tokens if not w in stop_words]
    result.append(temp)

dim = 3
model = Word2Vec(sentences=result, size=dim, window=5, min_count=5, workers=4, sg=1)

a=model.wv.most_similar("movie")
words = list(model.wv.vocab)
print('Vocabulary size : %d'%len(words))

model.wv.save_word2vec_format('word2vec.txt', binary=False)

embedding = {}
f = open('word2vec.txt', encoding='utf-8')

for line in f:
    wv = line.split()
    word = wv[0]
    vec = np.asarray(wv[1:])
    embedding[word] = vec
f.close

obj = Tokenizer()
obj.fit_on_texts(result)
seq = obj.texts_to_sequences(result)

word_index = obj.word_index
print('Found %s unique tokens.'%len(word_index))

review_pad = pad_sequences(seq, maxlen=max_length)
sentiment = df['tag'].values
print('Shape of review tensor : ', review_pad.shape)
print('Shape of sentiment tensor : ', sentiment.shape)

num_words = len(word_index) + 1
em_mat = np.zeros((num_words, dim))

for word, i in word_index.items():
    if i > num_words:
        continue
    embedding_vector = embedding.get(word)
    if embedding_vector is not None:
        em_mat[i] = embedding_vector
print(num_words)

model = Sequential()
embedding_layer = Embedding(num_words, dim, embeddings_initializer=Constant(em_mat),
                            input_length=max_length, trainable=False)
model.add(embedding_layer)
model.add(Bidirectional(CuDNNGRU(units=32, return_sequences=True)))
model.add(Bidirectional(CuDNNGRU(units=16, return_sequences=True)))
model.add(Bidirectional(CuDNNGRU(units=8, return_sequences=False)))
model.add(Dense(1, activation='relu'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

val_split = 0.2
indices = np.arange(review_pad.shape[0])
np.random.shuffle(indices)
review_pad = review_pad[indices]
sentiment = sentiment[indices]
num_val_samples = int(val_split * review_pad.shape[0])

x_train_pad = review_pad[:-num_val_samples]
y_train = sentiment[:-num_val_samples]
x_test_pad = review_pad[-num_val_samples:]
y_test = sentiment[-num_val_samples:]

print('Shape of x_train_pad tensor', x_train_pad.shape)
print('Shape of y_train tensor', y_train.shape)
print('Shape of x_test_pad tensor', x_test_pad.shape)
print('Shape of y_test tensor', y_test.shape)

print('Train....')
model.fit(x_train_pad, y_train, batch_size=64, epochs=50, validation_data=(x_test_pad, y_test), verbose = 2)

keys = ['price', 'samsung', 'asus', 'cpu', 'gpu', 'dell', 'lenovo', 'ram', 'weight', 'chromebook', 'sound', 'display', 'fan', 'battery']

embedding_clusters = []
word_clusters = []
for word in keys:
    embeddings = []
    words = []
    for similar_word, _ in model.most_similar(word, topn=30):
        words.append(similar_word)
        embeddings.append(model[similar_word])
    embedding_clusters.append(embeddings)
    word_clusters.append(words)

from sklearn.manifold import TSNE
import numpy as np

embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape
tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()


tsne_plot_similar_words('Similar words from Amazon', keys, embeddings_en_2d, word_clusters, 0.7,
                        'similar_words.png')

words_wp = []
embeddings_wp = []
for word in list(model.wv.vocab):
    embeddings_wp.append(model.wv[word])
    words_wp.append(word)

tsne_wp_3d = TSNE(perplexity=30, n_components=3, init='pca', n_iter=3500, random_state=12)
embeddings_wp_3d = tsne_wp_3d.fit_transform(embeddings_wp)

from mpl_toolkits.mplot3d import Axes3D


def tsne_plot_3d(title, label, embeddings, a=1):
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = cm.rainbow(np.linspace(0, 1, 1))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=colors, alpha=a, label=label)
    plt.legend(loc=4)
    plt.title(title)
    plt.show()

tsne_plot_3d('Visualizing Embeddings using t-SNE', ' word2vec', embeddings_wp_3d, a=0.1)
