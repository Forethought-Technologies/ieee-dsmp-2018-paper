from textblob.classifiers import NaiveBayesClassifier as NBC
from textblob.classifiers import DecisionTreeClassifier
from sklearn.metrics import f1_score
from utils import *

def naive_bayes_classify(data):
  class_to_predict = 'type' # product importance
  all_data = [tuple(x) for x in data[['text', class_to_predict]].to_records(index=False)]

  text_counts = {}
  for item in all_data:
    for word in set(item[0].split()):
      if word in text_counts:
        text_counts[word] += 1
      else:
        text_counts[word] = 1

  for i in range(len(all_data)):
    new_text = ''
    for word in all_data[i][0].split():
      if text_counts[word] >= 5:
        new_text += ' ' + word
    all_data[i] = (new_text, all_data[i][1])

  print('Finished preprocessing!')

  test_corpus = all_data[3000:3600]
  training_corpus = all_data[:3000]

  model = NBC(training_corpus, verbose=True)
  print('Done training!')
  print('Accuracy: ' + str(model.accuracy(test_corpus)))

  y_pred = []
  y_true = []
  for test_item in test_corpus:
    y_pred.append(model.prob_classify(test_item[0]).max())
    y_true.append(test_item[1])
  
  print('F1 score: ' + str(f1_score(y_true, y_pred, average='weighted')))

if __name__ == '__main__':
  print('Loading data!')
  data = load_chromium_bug_data()
  print('Classifying with NaiveBayes!')
  naive_bayes_classify(data)
  print('Done!')
