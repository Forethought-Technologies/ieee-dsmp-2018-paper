import pandas as pd
from text_processing import *

LINUX_BUGS_DATA_PATH = './linux_bugs_usage_ready.csv'
CHROMIUM_BUGS_DATA_PATH = './chromium.csv'

def merge_title_and_message(data, message_col_name='message'):
  '''
  This function is specific to the linux bug tracker dataset. It contains two
  feature columns (with text) - `title` and `message`, this merges them into a
  single column called `text`
  '''
  data['text'] = data['title'] + ' ' + data[message_col_name]
  data = data.drop(['title'], axis=1)
  data = data.drop([message_col_name], axis=1)
  return data

def load_linux_bug_data():
  '''
  Load linux bugs dataset and apply the preprocessing pipeline.
  '''
  data = pd.read_csv(LINUX_BUGS_DATA_PATH, sep='\t')
  data = merge_title_and_message(data)
  data = strip_punctuations(data)
  # data = stemm_text(data) - this has shown poor results
  data = remove_linux_garbage(data)
  return data

def load_chromium_bug_data():
  '''
  Load chromium bugs dataset and apply the preprocessing pipeline.
  '''
  data = pd.read_csv(CHROMIUM_BUGS_DATA_PATH, sep='\t')
  data = merge_title_and_message(data, message_col_name='description')
  data = strip_punctuations(data)
  return data
