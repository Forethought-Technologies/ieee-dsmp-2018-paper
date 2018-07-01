'''
This file contains miscellaneous utilities for text processing.
'''
import string
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

def strip_punctuations(data, column_name='text'):
  '''
  Strips punctuations from the end of each token.
  This uses suggestion from https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate
  to accomplish this really fast.
  '''
  translator = str.maketrans('', '', string.punctuation)
  data['text'] = data['text'].map(lambda s : str(s).translate(translator))
  return data

def stemm_text(data, stemmer_choice='Lancaster'):
  '''
  Stemm the 'text' column of data - this simplifies the words so
  that different forms of the same word end up being the same.
  '''
  if stemmer_choice == 'Lancaster':
    stemmer = LancasterStemmer()
  elif stemmer_choice == 'Snowball':
    stemmer = SnowballStemmer('english')
  elif stemmer_choice == 'Porter':
    stemmer = PorterStemmer()
  else:
    raise Exception('Illegal stemmer_choice argument')
  data['text'] = data['text'].map(lambda s : ' '.join(map(lambda w: stemmer.stem(w), s.split())))
  return data

def remove_long_words(data, maxlen=16):
  data['text'] = data['text'].map(lambda s : ' '.join(map(lambda w: w if len(w) < maxlen else ' ', s.split())))
  return data

def remove_short_words(data, minlen=4):
  data['text'] = data['text'].map(lambda s : ' '.join(map(lambda w: w if len(w) >= minlen else ' ', s.split())))
  return data

def remove_linux_garbage(data):
  '''
  Linux data contains lots of garbage, e.g. memory addresses - 0000f800
  '''
  def is_garbage(w):
    return len(w) >= 7 and sum(c.isdigit() for c in w) >= 2

  data['text'] = data['text'].map(lambda s : ' '.join(map(lambda w: w if not is_garbage(w) else ' ', s.split())))
  return data

def cast_to_lowercase(data):
  data['text'] = data['text'].map(lambda s : s.lower())
  return data

def remove_stopwords(data):
  stop_words = stopwords.words('english')
  translator = str.maketrans('', '', string.punctuation)
  stop_words = set([w.translate(translator) for w in stop_words]) # Apostrophes were removed already

  data['text'] = data['text'].map(lambda s : ' '.join(map(lambda w: w if w not in stop_words else ' ', s.split())))
  return data

def remove_rare_words(data, min_count=3):
  wc = {} # WordCount
  def proc_word(s):
    for w in set(s.split()):
      if w in wc:
        wc[w] += 1
      else:
        wc[w] = 1

  for index, row in data.iterrows():
    proc_word(row['text'])

  data['text'] = data['text'].map(lambda s : ' '.join(map(lambda w: w if wc[w] >= min_count else ' ', s.split())))
  return data
