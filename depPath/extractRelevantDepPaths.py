import os
import pprint
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import Counter

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()
parser.add_argument('--wikideppaths', type=str, required=True)
parser.add_argument('--trfile', type=str, required=True)

parser.add_argument('--outputfile', type=str, required=True)


def extractRelevantPaths(wikideppaths, wordpairs_labels, outputfile):
    '''Each line in wikideppaths contains 3 columns
        col1: word1
        col2: word2
        col3: deppath
    '''

    print(wikideppaths)

    lines_read = 0
    relevantDepPaths2counts = Counter()
    word_pair_counts = Counter() 
    path2direct = {}
    with open(wikideppaths, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            lines_read += 1

            word1, word2, deppath = line.split("\t")
            if (word1, word2) in wordpairs_labels:       
                if wordpairs_labels[(word1, word2)] == "True":
                    word_pair_counts[(word1, word2)] = word_pair_counts[(word1, word2)] + 1
                    # categorize as forward
                    if deppath not in path2direct: 
                        path2direct[deppath] = 'forward'
                    # increment relevant path counts 
                    relevantDepPaths2counts[deppath] = relevantDepPaths2counts[deppath] + 1
            elif (word2, word1) in wordpairs_labels:
                if wordpairs_labels[(word2, word1)] == "True": 
                    # categorize as reverse
                    word_pair_counts[(word2, word1)] = word_pair_counts[(word1, word2)] + 1
                    if deppath not in path2direct: 
                        path2direct[deppath] = 'reverse'
                        # increment relevant path counts
                    relevantDepPaths2counts[deppath] = relevantDepPaths2counts[deppath] + 1
            '''
                IMPLEMENT METHOD TO EXTRACT RELEVANT DEPEDENCY PATHS HERE

                Make sure to be clear about X being a hypernym/hyponym.

                Dependency Paths can be extracted in multiple different categories, such as
                1. Forward Paths: X is hyponym, Y is hypernym
                2. Reverse Paths: X is hypernym, Y is hyponym
                3. Negative Paths: If this path exists, definitely not a hyper/hyponym relations
                4. etc......
            '''
    counts = relevantDepPaths2counts.values()
    counts = list(counts)
    print(type(counts))
    print(max(counts))
    print(min(counts))
    counts = np.asarray(counts)
    print(np.percentile(counts, 10))
    print(np.percentile(counts, 20))
    print(np.percentile(counts, 30))
    print(np.percentile(counts, 40))
    print(np.percentile(counts, 50))
    print(np.percentile(counts, 60))
    print(np.percentile(counts, 70))
    print(np.percentile(counts, 80))
    print(np.percentile(counts, 90))
    print(np.percentile(counts, 95))
    print(np.percentile(counts, 100))


    word_counts = word_pair_counts.values()
    word_counts = list(word_counts)
    print(type(word_counts))
    print(max(word_counts))
    print(min(word_counts))
    word_counts = np.asarray(word_counts)
    print(np.percentile(word_counts, 10))
    print(np.percentile(word_counts, 20))
    print(np.percentile(word_counts, 30))
    print(np.percentile(word_counts, 40))
    print(np.percentile(word_counts, 50))
    print(np.percentile(word_counts, 60))
    print(np.percentile(word_counts, 70))
    print(np.percentile(word_counts, 80))
    print(np.percentile(word_counts, 90))
    print(np.percentile(word_counts, 95))
    print(np.percentile(word_counts, 100))


    toppaths = relevantDepPaths2counts.most_common()[1:20]
    for path, count in toppaths:
        print(path, " ", str(count), " ", str(path2direct[path]))
    with open(outputfile, 'w') as f:
        for dep_path in relevantDepPaths2counts:
            if relevantDepPaths2counts[dep_path] > 1:
                f.write(dep_path)
                f.write('\t')
                f.write(path2direct[dep_path])
                f.write('\n')


def readVocab(vocabfile):
    vocab = set()
    with open(vocabfile, 'r') as f:
        for w in f:
            if w.strip() == '':
                continue
            vocab.add(w.strip())
    return vocab


def readWordPairsLabels(datafile):
    wordpairs = {}
    with open(datafile, 'r') as f:
        inputdata = f.read().strip()

    inputdata = inputdata.split("\n")
    for line in inputdata:
        word1, word2, label = line.strip().split('\t')
        word1 = word1.strip()
        word2 = word2.strip()
        wordpairs[(word1, word2)] = label
    return wordpairs


def main(args):
    print(args.wikideppaths)

    wordpairs_labels = readWordPairsLabels(args.trfile)
    print(len(wordpairs_labels))
    print(wordpairs_labels[("guitar", "flute")])
    print(wordpairs_labels[("rifle", "gun")])
    print(wordpairs_labels[("guitar", "banjo")])

    print("Total Number of Word Pairs: {}".format(len(wordpairs_labels)))

    extractRelevantPaths(args.wikideppaths, wordpairs_labels, args.outputfile)


if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
