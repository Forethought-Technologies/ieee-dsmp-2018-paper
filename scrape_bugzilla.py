'''
This code scrapes data from the linux bugtracker (e.g.
    https://bugzilla.kernel.org/show_bug.cgi?id=1)
and saves the results as csv that contains the following columns:

* Title
* Message
* Importance (e.g. P2 normal)
* Product
* Component

All of these fields are text - the last 3 can be converted to classes by the user
of the csv.

Parsing is done with BeautifulSoup library (`pip install bs4` to get it).
'''
import os
import time
import csv
from bs4 import BeautifulSoup
import urllib.request

def create_csv(filename, fieldnames, data):
  '''
  Creates a CSV file with the provided fieldnames and data, where data
  is the list of dictionaries with field values.
  '''
  with open(filename, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, delimiter='\t', fieldnames=fieldnames)
    writer.writeheader()
    # Write out each row. row is exactly the dict that DictWriter expects.
    for row in data:
      writer.writerow(row)

def scrape_page(bug_idx):
  '''
  Scrapes one page of bugzilla and returns a dictionary.
  '''
  BASE_BUG_URL = 'https://bugzilla.kernel.org/show_bug.cgi?id='
  bug_url = BASE_BUG_URL + str(bug_idx)

  bug_page = urllib.request.urlopen(bug_url).read()
  soup = BeautifulSoup(bug_page)

  # Parse meaningful attributes
  title = soup.title.text
  title = ' '.join(title.split(' – ')[1:]) # Remove the bug number from the title

  message = ' '.join(soup.find('div', {'id': 'c0'}).find('pre').text.split())

  # Some ugly parsing to get the content and remove some redundant parts.
  importance = ' '.join(soup.find('tr', {'id':
    'field_tablerow_importance'}).find('td').string.split())
  product = soup.find('td', {'id': 'field_container_product'}).text.strip()
  component = soup.find('td', {'id': 'field_container_component'}).text.split(
    '(show')[0].strip()

  return {
    'title': title,
    'message': message,
    'importance': importance,
    'product': product,
    'component': component
  }

def scrape_bugzilla():
  FIELD_NAMES = ['title', 'message', 'importance', 'product', 'component']
  all_results = []
  for bug_idx in range(1, 16561): # Magic number of bugs in the tracker
    if bug_idx % 100 == 0:
      print('At index ' + str(bug_idx) + ' ; parsed so far ' + str(len(all_results)))
    try:
      result = scrape_page(bug_idx)
    except:
      continue
    all_results.append(result)
    #time.sleep(1) # Wait to avoid triggering anti-DOS
  create_csv('linux_bugs.csv', FIELD_NAMES, all_results)

if __name__ == '__main__':
  scrape_bugzilla()
