import os
import pprint
import argparse

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--wikideppaths', type=str, required=True)
parser.add_argument('--relevantdeppaths', type=str, required=True)
parser.add_argument('--outputfile', type=str, required=True)


def extractHyperHypoExtractions(wikideppaths, relevantPaths):
    '''Each line in wikideppaths contains 3 columns
        col1: word1
        col2: word2
        col3: deppath
    '''
    
    '''
        IMPLEMENT
    '''

    # Should finally contain a list of (hyponym, hypernym) tuples
    depPathExtractions = []   
    with open(wikideppaths, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            word1, word2, deppath = line.split("\t")
            if deppath in relevantPaths: 
                path_type = relevantPaths[deppath].rstrip()
                if path_type == 'forward':
                    depPathExtractions.append((word1, word2))
                elif path_type == 'reverse':
                    depPathExtractions.append((word2, word1))
    return depPathExtractions


def readPaths(relevantdeppaths):
    '''
        READ THE RELEVANT DEPENDENCY PATHS HERE
    '''
    path2type = {}
    with open(relevantdeppaths, 'r') as f: 
        lines = f.readlines() 
        for line in lines:
            splitline = line.split('\t')
            path = splitline[0]
            path_type = splitline[1]
            path2type[path] = path_type
    print("Relevant Paths: " + str(len(path2type)))
    return path2type        
    # return relevantPaths


def writeHypoHyperPairsToFile(hypo_hyper_pairs, outputfile):
    directory = os.path.dirname(outputfile)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(outputfile, 'w') as f:
        for (hypo, hyper) in hypo_hyper_pairs:
            f.write(hypo + "\t" + hyper + '\n')


def main(args):
    print(args.wikideppaths)

    relevantPaths = readPaths(args.relevantdeppaths)
    # relevant path is a dictionary where paths are keys and 
    # types of paths are values 
    hypo_hyper_pairs = extractHyperHypoExtractions(args.wikideppaths,
                                                   relevantPaths)
    print(len(hypo_hyper_pairs))
    writeHypoHyperPairsToFile(hypo_hyper_pairs, args.outputfile)


if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
