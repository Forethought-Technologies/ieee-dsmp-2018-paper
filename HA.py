import re
from itertools import chain
import string


import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras import backend as K

from keras.layers import Dense, Input
from keras.layers import GRU, Bidirectional, TimeDistributed, CuDNNLSTM, LSTM, Dropout, CuDNNGRU
from keras.models import Model
from keras.optimizers import  RMSprop
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from attention_layer import AttentionWithContext
from utils import merge_title_and_message, remove_linux_garbage, remove_stopwords

def read_linux(feature):
    data = pd.read_csv('./linux_bugs_usage_ready.csv', sep='\t')
    data = merge_title_and_message(data)
    data = remove_linux_garbage(data)

    return data['text'], data[feature]

def read_chrome():
    data = pd.read_csv('./chromium.csv', sep='\t')
    data = merge_title_and_message(data, message_col_name='description')
    # data = remove_linux_garbage(data)
    data['text'] = data['text'].map(lambda s: str(s).replace('\\r', '').replace('\\n', '. '))
    return data['text'], data['type']

X, Y = read_chrome() # read_linux('importance')

RNN = CuDNNGRU

enc = LabelEncoder()
yc = enc.fit_transform(Y)
oh = LabelBinarizer()
y_trans = oh.fit_transform(yc)

translator = str.maketrans('', '', string.punctuation)
stop_words = stopwords.words('english')
stop_words = set([w.translate(translator) for w in stop_words])

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    string = string.strip().lower().translate(translator)
    return string

def remove_stopwords_from_sent(sent):
    res = []
    for word in sent:
        if word not in stop_words:
            res.append(word)
    return res


def build_sentences(X):
    X_sentences = []
    for doc in X:
        sentences = sent_tokenize(doc)
        cleaned = map(clean_str, sentences)
        tokenized = map(word_tokenize, cleaned)
        cleaned = map(remove_stopwords_from_sent, tokenized)
        X_sentences.append(list(cleaned))

    return X_sentences

X_sentences = build_sentences(X)

list(map(print, X_sentences[2]))


# Word2vec parameters
min_word_frequency_word2vec = 3
embed_size_word2vec = 200
context_window_word2vec = 5

X_merged = list(map(lambda l: list(chain(*l)), X_sentences))

print(X_merged[13])

wordvec_model = Word2Vec(X_merged, min_count=min_word_frequency_word2vec,
                         size=embed_size_word2vec, window=context_window_word2vec)
max_doc_len = 5
max_sentence_len = 100
num = len(X_sentences)

vocabulary = wordvec_model.wv.vocab
print("Vocabulary", len(vocabulary))


def map_sentence(sent):
    out = np.empty((max_sentence_len, embed_size_word2vec))
    for ind, word in enumerate(sent):
        if ind == max_sentence_len:
            break
        if word in vocabulary:
            out[ind, :] = wordvec_model.wv[word]
    return out


def map_doc(doc):
    out = np.empty((max_doc_len, max_sentence_len, embed_size_word2vec))
    for ind, sent in enumerate(doc):
        if ind == max_doc_len:
            break
        out[ind, :] = map_sentence(sent)
    return out


x = np.empty((num, max_doc_len, max_sentence_len, embed_size_word2vec))

for ind, doc in enumerate(X_sentences):
    x[ind, :] = map_doc(doc)

def make_model(rnn_dim=64, dense_dim=50):

    def attention_block():
        def f(input):
            rnn = Bidirectional(RNN(rnn_dim, return_sequences=True))(input)
            drop1 = Dropout(0.75)(rnn)
            dense = TimeDistributed(Dense(dense_dim))(drop1)
            drop2 = Dropout(0.6)(dense)
            att = AttentionWithContext()(drop2)
            return att
        return f

    with K.name_scope('sentence_enc'):
        sentence_input = Input(shape=(max_sentence_len, embed_size_word2vec))
        word_att = attention_block()(sentence_input)
        sentEncoder = Model(sentence_input, word_att)

    with K.name_scope('doc_enc'):
        doc_input = Input(shape=(max_doc_len, max_sentence_len, embed_size_word2vec))
        sent_enc = TimeDistributed(sentEncoder)(doc_input)
        doc_att = attention_block()(sent_enc)
        preds = Dense(y_trans.shape[-1], activation='softmax')(doc_att)

        model = Model(doc_input, preds)
        return model

model = make_model(rnn_dim=64, dense_dim=64)
model.summary()

sz = len(y_trans)
x_train = x[int(0.1 * sz):int(0.95 * sz)]
x_test = np.stack(x[:int(0.1 * sz)], x[int(0.95 * sz):])

y_train = y_trans[int(0.1 * sz):int(0.95 * sz)]
y_test = np.stack(y_trans[:int(0.1 * sz)], y_trans[int(0.95 * sz):])

#x_train, x_test, y_train, y_test = train_test_split(x, y_trans, train_size=0.85)


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train, validation_data=(x_test, y_test),
          nb_epoch=10, batch_size=16)


from sklearn.metrics import accuracy_score, f1_score

def report(x, y):
    labels = np.argmax(y, axis=-1)
    predicted = np.argmax(model.predict(x), axis=-1)

    print("Acc", accuracy_score(labels, predicted))
    print("F1", f1_score(labels, predicted, average='weighted'))

print("Training")
report(x_train, y_train)

print("Testing")
report(x_test, y_test)
