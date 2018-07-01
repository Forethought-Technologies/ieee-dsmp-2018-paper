from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
import numpy as np
from utils import *
from grid_search import *
from text_processing import *

def tfidf_classify(data, model_type='SVM', extra_params={'min_df': 0.001}):
  '''
  data is a pandas dataframe
  '''
  class_to_predict = 'type' # product importance
  data = shuffle(data, random_state=77)

  num_records = len(data)
  data_train = data[:int(0.85 * num_records)]
  data_test = data[int(0.85 * num_records):]

  train_data = [x[0] for x in data_train[['text']].to_records(index=False)]
  train_labels = [x[0] for x in data_train[[class_to_predict]].to_records(index=False)]

  test_data = [x[0] for x in data_test[['text']].to_records(index=False)]
  test_labels = [x[0] for x in data_test[[class_to_predict]].to_records(index=False)]

  # Create feature vectors 
  vectorizer = TfidfVectorizer(**extra_params)
  # Train the feature vectors
  train_vectors = vectorizer.fit_transform(train_data)
  test_vectors = vectorizer.transform(test_data)

  # Perform classification with SVM, kernel=linear 
  if model_type == 'SVM':
    model = svm.SVC(kernel='linear')
  elif model_type == 'NN':
    model = MLPClassifier(hidden_layer_sizes=(200, 200), max_iter=4000)
  print('Training the model!')
  model.fit(train_vectors, train_labels) 
  train_prediction = model.predict(train_vectors)
  test_prediction = model.predict(test_vectors)

  train_accuracy = np.sum((np.array(train_labels) == np.array(train_prediction))) * 1.0 / len(train_labels)
  print('Training accuracy: ' + str(train_accuracy))

  test_accuracy = np.sum((np.array(test_labels) == np.array(test_prediction))) * 1.0 / len(test_labels)
  print('Test accuracy: ' + str(test_accuracy))

  print('F1 score: ' + str(f1_score(test_labels, test_prediction, average='weighted')))

  return test_accuracy

if __name__ == '__main__':
  print('Loading data!')
  data = load_chromium_bug_data()

  # Additional steps of the pipeline - FastText can do it by itself, or it doesn't help it
  data = cast_to_lowercase(data)
  data = remove_stopwords(data)
  data = remove_rare_words(data, min_count=3)

  print('Classifying with TFIDF-based approach!')

  tfidf_classify(data, model_type='SVM')

  #param_options = {
  #  'min_df' : [0, 0.001, 0.01],
  #  'max_df' : [0.5, 0.8, 1.0]
  #}

  #grid_search(tfidf_classify, data, param_options)
  print('Done!')
