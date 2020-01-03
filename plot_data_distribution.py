import pandas as pd
import gzip
import os
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import dump_svmlight_file
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)
#JSON文件生成pands data frame
def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


def get_path(folder_name, file_list):
  file_name = os.listdir(folder_name)
  for file in file_name:
    file_path = os.path.join(folder_name, file)
    if os.path.isfile(file_path):
      file_list.append(file_path)
  return file_name

if __name__ == '__main__':
   data_name = "Tool"
   data_path = os.path.join('./Source/', data_name+".gz")
   df = getDF(data_path)
   doc = df.reviewText
   label = df.overall
   result = pd.value_counts(label)
   index = []
   num = []
   for idx in result.index:
       index.append(int(idx))
       num.append(result[idx])
       print(result[idx])
   plt.bar(index, num, label="number")
   plt.legend()
   plt.xlabel('labels')
   plt.ylabel('numbers')
   plt.show()
