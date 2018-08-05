'''
FastText requires specific training file format - see
https://github.com/facebookresearch/fastText for details.
'''
import fastText
from utils import *
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from grid_search import *

TRAIN_PATH = './fasttext_train.txt'
TEST_PATH = './fasttext_test.txt'
MODEL_PATH = './ft.model'

def fasttext_classify(data, extra_params={}):
  class_to_predict = 'type' # product importance
  data[class_to_predict] = data[class_to_predict].map(lambda s : s.replace(" ", ""))
  data_for_fasttext = data['text'] + ' __label__' + data[class_to_predict]
  data_for_fasttext = shuffle(data_for_fasttext, random_state=77)

  num_records = len(data_for_fasttext)
  data_train = data_for_fasttext[:int(0.85 * num_records)]
  data_test = data_for_fasttext[int(0.85 * num_records):]

  data_train.to_csv(TRAIN_PATH, sep='\t', header=0, index=False)
  data_test.to_csv(TEST_PATH, sep='\t', header=0, index=False)

  model = fastText.train_supervised(TRAIN_PATH, **extra_params)
  #model.save_model(MODEL_PATH)
  print('Training accuracy:')
  train_accuracy = model.test(TRAIN_PATH)
  print(train_accuracy[-1])

  print('Test accuracy:')
  test_accuracy = model.test(TEST_PATH)
  print(test_accuracy[-1])

  y_pred = []
  y_true = []
  for test_item in data_test:
    test_text, test_label = test_item.split('__label__')
    y_pred.append(model.predict(test_text)[0])
    y_true.append('__label__' + test_label)

  print('F1 score: ' + str(f1_score(y_true, y_pred, average='weighted')))

  return test_accuracy[-1] # accuracy is a tuple

if __name__ == '__main__':
  print('Loading data!')
  data = load_chromium_bug_data()
  print('Classifying with FastText!')

  #param_options = {
  #  'epoch' : [200, 1000],
  #  'minCount' : [1, 3, 5, 10],
  #  'dim': [10, 100, 500, 1000, 4000],
  #  'ws': [3, 5, 7],
  #  'neg': [1, 5, 10],
  #  'wordNgrams': [1, 2, 5, 10],
  #  'verbose': [0]
  #}

  # These are the optimal parameters for the 'importance' prediction
  # For other columns they are different.
  fasttext_classify(data, extra_params={
    'epoch': 15,
    'minCount': 1,
    'dim': 150,
    'ws': 5,
    'neg': 5,
    'wordNgrams': 2,
    'verbose': 2
  })
  #grid_search(fasttext_classify, data, param_options)
  print('Done!')
