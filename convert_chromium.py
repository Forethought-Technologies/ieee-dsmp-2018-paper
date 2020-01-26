import os
import json
from scrape_bugzilla import create_csv

# Download from http://bugtriage.mybluemix.net/ - "Google Chromium"
DATASET_PATH = './chromium_all_data.json'

types = ['Bug', 'Feature', 'Compat']
counts = {
  'Bug': 0,
  'Feature': 0,
  'Compat': 0
}

all_data = []
for entry in json.load(open(DATASET_PATH), strict=False):
  if entry['type'] not in types:
    continue
  counts[entry['type']] += 1
  if counts[entry['type']] >= 30000:
    continue # Prevent imbalance because of number of bugs
  all_data.append({
    'title': entry['issue_title'].encode("unicode_escape").decode("utf-8"),
    'description': entry['description'].encode("unicode_escape").decode("utf-8"),
    'type': entry['type']
  })
FIELD_NAMES = ['title', 'description', 'type']
create_csv('chromium.csv', FIELD_NAMES, all_data)
