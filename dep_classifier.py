import re
import nltk
from nltk.tag.perceptron import PerceptronTagger
import numpy as np 
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def map_pairs_to_paths(wikideppaths):
  '''Each line in wikideppaths contains 3 columns
      col1: word1
      col2: word2
      col3: deppath
  '''
  pair_to_path_counts = {}
  with open(wikideppaths, 'r') as f:
    for line in f:
      line = line.strip()
      if not line:
          continue
      word1, word2, deppath = line.split("\t")
      word1 = word1.lower()
      word2 = word2.lower()
      if (word1, word2) not in pair_to_path_counts:
        pair_to_path_counts[(word1, word2)] = {}
      if (word2, word1) not in pair_to_path_counts: 
        pair_to_path_counts[(word2, word1)] = {}
      # Adding path
      if deppath not in pair_to_path_counts[(word1, word2)]: 
        pair_to_path_counts[(word1, word2)][deppath] = 1
      else: 
        pair_to_path_counts[(word1, word2)][deppath] = pair_to_path_counts[(word1, word2)][deppath] + 1
      # Adding reverse path
      if ("REVERSE" + deppath) not in pair_to_path_counts[(word2, word1)]: 
        pair_to_path_counts[(word2, word1)]["REVERSE" + deppath] = 1
      else: 
        pair_to_path_counts[(word2, word1)]["REVERSE" + deppath] = pair_to_path_counts[(word2, word1)]["REVERSE" + deppath] + 1
  return pair_to_path_counts  

def write_predictions_to_file(predname, preds, pairs):
  output_lines = []
  for pred, (hypo, hyper) in zip(preds, pairs):
    if pred == 0:
      output_lines.append('{}\t{}\tFalse'.format(hypo, hyper))
    else:
      output_lines.append('{}\t{}\tTrue'.format(hypo, hyper))
  outputfile = predname + "_pred.txt"
  with open(outputfile, 'w') as f: 
    for line in output_lines:
      f.write(line)
      f.write('\n')

def pop_training_data(trainingpath, pair_to_path_counts): 
  examples = []
  labels = []
  pairs = []

  with open(trainingpath, 'r') as f:
    inputdata = f.read().strip()
    inputdata = inputdata.split("\n")
    for line in inputdata:
      word1, word2, label = line.strip().split('\t')
      word1 = word1.strip()
      word2 = word2.strip()
      pairs.append((word1, word2))
      if label == "True":
        label = 1
      elif label == "False": 
        label = 0
      if (word1, word2) in pair_to_path_counts:
        examples.append(pair_to_path_counts[(word1, word2)])
        labels.append(label)
      else: 
        examples.append(dict())
        labels.append(label)
  return examples, labels, pairs

def pop_test_data(testpath, pair_to_path_counts):
  examples = []
  pairs = []

  with open(testpath, 'r') as f:
    inputdata = f.read().strip()
    inputdata = inputdata.split("\n")
    for line in inputdata:
      word1, word2 = line.strip().split('\t')
      word1 = word1.strip()
      word2 = word2.strip()
      pairs.append((word1, word2))
      if (word1, word2) in pair_to_path_counts:
        examples.append(pair_to_path_counts[(word1, word2)])
      else: 
        examples.append(dict())
  return examples, pairs

if __name__=='__main__':
  # wikideppaths file
  deppaths_file = "new_wikipedia_deppaths.txt"
  trainingfile = "data_lex_train.tsv"
  valfile = "data_lex_val.tsv"
  testfile = "data_lex_test.tsv"

  pairs_to_paths = map_pairs_to_paths(deppaths_file)
  x_train, train_labels, train_pairs = pop_training_data(trainingfile, pairs_to_paths)
  x_val, val_labels, val_pairs = pop_training_data(valfile, pairs_to_paths)
  x_test, test_pairs = pop_test_data(testfile, pairs_to_paths)

  print("Pairs and training data set up")
  vectorizer = DictVectorizer()

  x_train = vectorizer.fit_transform(x_train)
  x_val = vectorizer.transform(x_val)
  x_test = vectorizer.transform(x_test)

  print("Data Transformed")

  log1 = LogisticRegression(C = .01)
  log2 = LogisticRegression(C = .1)
  log3 = LogisticRegression(C = 1)
  log4 = LogisticRegression(C = 10)
  log5 = LogisticRegression(C = 100)

  print("Setup complete, training models...")

  log1.fit(x_train, train_labels)
  log2.fit(x_train, train_labels)
  log3.fit(x_train, train_labels)
  log4.fit(x_train, train_labels)
  log5.fit(x_train, train_labels)

  print("Training complete...forecasting")

  val1_pred = log1.predict(x_val)
  val2_pred = log2.predict(x_val)
  val3_pred = log3.predict(x_val)
  val4_pred = log4.predict(x_val)
  val5_pred = log5.predict(x_val)
  
  test1_pred = log1.predict(x_test)
  test2_pred = log2.predict(x_test)
  test3_pred = log3.predict(x_test)
  test4_pred = log4.predict(x_test)
  test5_pred = log5.predict(x_test)


  print("writing results") 

  write_predictions_to_file("val1", val1_pred, val_pairs)
  write_predictions_to_file("val2", val2_pred, val_pairs)
  write_predictions_to_file("val3", val3_pred, val_pairs)
  write_predictions_to_file("val4", val4_pred, val_pairs)
  write_predictions_to_file("val5", val5_pred, val_pairs)  

  write_predictions_to_file("test1", test1_pred, test_pairs)
  write_predictions_to_file("test2", test2_pred, test_pairs)
  write_predictions_to_file("test3", test3_pred, test_pairs)
  write_predictions_to_file("test4", test4_pred, test_pairs)
  write_predictions_to_file("test5", test5_pred, test_pairs)  

