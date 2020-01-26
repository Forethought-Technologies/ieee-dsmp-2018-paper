# This code is adapted from http://bugtriage.mybluemix.net/

import itertools
import numpy as np
np.random.seed(1337)
from gensim.models import Word2Vec
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, merge, BatchNormalization, Bidirectional
from keras.utils import np_utils
from keras.optimizers import RMSprop
from sklearn.utils import shuffle
from utils import *
from text_processing import *

# Word2vec parameters
min_word_frequency_word2vec = 3
embed_size_word2vec = 200
context_window_word2vec = 5

# Classifier hyperparameters
max_sentence_len = 100 # 9000
min_sentence_length = 2
batch_size = 32


def bidir_rnn_classify(data):
  '''
  Use the approach suggested at http://bugtriage.mybluemix.net/ to do
  classification.
  '''
  class_to_predict = 'importance' # component product importance
  data = shuffle(data, random_state=77)

  num_records = len(data)
  data_train = data[:int(0.85 * num_records)]
  data_test = data[int(0.85 * num_records):]

  train_data = [x[0] for x in data_train[['text']].to_records(index=False)]
  train_labels = [x[0] for x in data_train[[class_to_predict]].to_records(index=False)]
  unique_train_label = list(set(train_labels))

  test_data = [x[0] for x in data_test[['text']].to_records(index=False)]
  test_labels = [x[0] for x in data_test[[class_to_predict]].to_records(index=False)]

  # Tokenize data
  train_data = [text.split() for text in train_data] # TODO(Vladimir) - try nltk tokenize here
  test_data = [text.split() for text in test_data]
  all_data = train_data + test_data

  print('Data examples')
  print(all_data[:5])

  # Generate word2vec
  wordvec_model = Word2Vec(all_data, min_count=min_word_frequency_word2vec,
                           size=embed_size_word2vec, window=context_window_word2vec)
  vocabulary = wordvec_model.wv.vocab
  vocab_size = len(vocabulary)

  print('Vocab size is ' + str(vocab_size))

  X_train = np.empty(shape=[len(train_data), max_sentence_len, embed_size_word2vec],
                     dtype='float32')
  Y_train = np.empty(shape=[len(train_labels), 1], dtype='int32')
  # 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3
  print('Building X_train!')
  for j, curr_row in enumerate(train_data):
    if j % 100 == 0:
      print('Building X_train j = ' + str(j))
    sequence_cnt = 0
    for item in curr_row:
      if item in vocabulary:
        X_train[j, sequence_cnt, :] = wordvec_model[item]
        sequence_cnt = sequence_cnt + 1
        if sequence_cnt == max_sentence_len - 1:
          break
    for k in range(sequence_cnt, max_sentence_len):
      X_train[j, k, :] = np.zeros((1, embed_size_word2vec))
    Y_train[j, 0] = unique_train_label.index(train_labels[j])

  X_test = np.empty(shape=[len(test_data), max_sentence_len, embed_size_word2vec],
                    dtype='float32')
  Y_test = np.empty(shape=[len(test_labels), 1], dtype='int32')
  # 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3
  print('Building X_test!')
  for j, curr_row in enumerate(test_data):
    if j % 100 == 0:
      print('Building X_test j = ' + str(j))
    sequence_cnt = 0
    for item in curr_row:
      if item in vocabulary:
        X_test[j, sequence_cnt, :] = wordvec_model[item]
        sequence_cnt = sequence_cnt + 1
        if sequence_cnt == max_sentence_len - 1:
          break
    for k in range(sequence_cnt, max_sentence_len):
      X_test[j, k, :] = np.zeros((1, embed_size_word2vec))
    Y_test[j, 0] = unique_train_label.index(test_labels[j])

  y_train = np_utils.to_categorical(Y_train, len(unique_train_label))

  print('Bulding KERAS models!')

  sequence = Input(shape=(max_sentence_len, embed_size_word2vec), dtype='float32')
  lstm = Bidirectional(LSTM(1024))(sequence)
  after_dp = Dropout(0.5)(lstm)
  output = Dense(len(unique_train_label), activation='softmax')(after_dp)
  model = Model(input=sequence, output=output)
  rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
  model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])

  # Train the model
  print('Training the model!')
  hist = model.fit(X_train, y_train,
                   batch_size=batch_size,
                   nb_epoch=5)

  train_result = hist.history
  print(train_result)

  train_prediction = model.predict(X_train)

  total_train_correct = 0
  for j, ll in enumerate(train_prediction):
    if np.argmax(ll) == Y_train[j]:
      total_train_correct += 1

  print('Train accuracy:', total_train_correct * 1.0 / len(train_prediction))

  test_prediction = model.predict(X_test)

  total_test_correct = 0
  labels = []
  predicted = []
  for j, ll in enumerate(test_prediction):
    predicted.append(np.argmax(ll))
    labels.append(Y_test[j])
    if np.argmax(ll) == Y_test[j]:
      total_test_correct += 1

  print('Test accuracy:', total_test_correct * 1.0 / len(test_prediction))
  print('Test F1:', f1_score(labels, predicted, average='weighted'))

  return total_test_correct * 1.0 / len(test_prediction)

if __name__ == '__main__':
  print('Loading data!')
  data = load_linux_bug_data()
  data = cast_to_lowercase(data)
  data = remove_stopwords(data)
  print('Classifying with bidirectional RNNs!')
  bidir_rnn_classify(data)
  print('Done!')
