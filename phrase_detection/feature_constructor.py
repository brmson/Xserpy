""""Construct features for phrase detection"""
import os
import argparse
import json
import pickle
import nltk
from nltk import StanfordNERTagger

from phrase_detection.phrase_detector import predict


class Question(object):
    """Class representing a question from the Free917 dataset"""
    def __init__(self,  utterance,  targetFormula):
        self.utterance = utterance
        self.targetFormula = targetFormula

def object_decoder(obj):
    """Hook a json data to object"""
    return Question(obj['utterance'],  obj['targetFormula'])

def uni_bi_tri(c, u, j):
    """Construct a uni-, bi- and trigram feature from given data

    Keyword arguments:
    c -- type of feature (p, w, n)
    u -- list of words or tags
    j -- index of checked word

    """
    feature = []
    # Unigram
    feature.append(c+"_u."+u[j])

    # Bigram
    feature.append(c+"_b0."+u[j-1]+"_"+u[j])
    feature.append(c+"_b1."+u[j]+"_"+u[j+1])

    # Trigram
    feature.append(c+"_t0."+u[j-2]+"_"+u[j-1]+"_"+u[j])
    feature.append(c+"_t1."+u[j-1]+"_"+u[j]+"_"+u[j+1])
    feature.append(c+"_t2."+u[j]+"_"+u[j+1]+"_"+u[j+2])

    return feature

def sub2ind(x, y, z, l):
    """Convert indexes of 3-dimensional (cubic) array to 1-dimensional array

    Keyword arguments:
    x -- x coordinate
    y -- y coordinate
    z -- z coordinate
    l -- size of one dimension of the 3D array

    """
    return x + l * (y + l * z)

def ner_tag(questions, path, java_path):
    """Tag each word in given set of questions with NER tag then return list of lists of tags

    Keyword arguments:
    questions -- list of Question objects
    path -- a path to Stanford NLP library
    java_path -- path to Java executable

    """

    sep = os.path.sep
    # Uses Stanford NER tagger with a dictionary
    st_ner = StanfordNERTagger(path+"classifiers" + sep + "english.all.3class.distsim.crf.ser.gz", path+"stanford-ner.jar")
    os.environ['JAVAHOME'] = java_path

    tagged = []
    for q in questions:
        text = nltk.word_tokenize(q.utterance)
        tagged.append(st_ner.tag(text))
    return tagged

def pos_tag(questions):
    """Tag each word in given set of questions with POS tag then return list of lists of tags

    Keyword arguments:
    questions -- list of Question objects

    """
    tagged = []
    i = 0
    for q in questions:
        text = nltk.word_tokenize(q.utterance)
        tagged.append(nltk.pos_tag(text))
        i += 1
    return tagged

def construct_feature(p, u, n, j, l):
    """Construct a feature vector (represented as array of strings) of a word used in phrase detection

    Keyword arguments:
    p -- list of POS tags for corresponding words in question
    u -- list of words in question
    n -- list of NER tags for corresponding words in question
    j -- position of the word in question utterance
    l -- label of the word preceding processed word

    """
    w_f = uni_bi_tri('w', u, j)
    p_f = uni_bi_tri('p', p, j)
    n_f = uni_bi_tri('n', n, j)

    feature = w_f + p_f + n_f

    feature.append("l."+str(l))
    feature.append("l_w."+str(l)+"_"+u[j])

    return feature

def label_phrases(questions, pos_tagged, ner_tagged, weights):
    """Find phrase labels for words without knowing gold standard labels

    Keyword arguments:
    questions -- list of Question objects
    pos_tagged -- list of lists of POS tagged words
    ner_tagged -- list of lists of NER tagged words
    weights -- model trained for phrase detection

    """
    features = []
    labels = []
    for i in range(len(questions)):
        l = 4
        question = questions[i]
        u = ["", ""]+question.utterance.split()+["", ""]
        p = ['', '']+[pp[1] for pp in pos_tagged[i]]+['', '']
        n = ['', '']+[nn[1] for nn in ner_tagged[i]]+['', '']
        for j in range(2, len(u)-2):
            f = construct_feature(p, u, n, j, l)
            l = predict(weights, f, 5)
            features.append(f)
            labels.append(l)
    return features, labels

def create_features(questions, pos_tagged, ner_tagged, path):
    """Create feature vectors for all questions in a dataset

    Keyword arguments:
    questions -- list of Question objects
    pos_tagged -- list of lists of POS tagged words
    ner_tagged -- list of lists of NER tagged words
    path -- path to gold standard phrase labels

    """
    labels = pickle.load(open(path))

    features = []
    for i in range(len(questions)):

        question = questions[i]

        # Each list is expanded in order to be possible to look at tags/words beyond utterance span
        u = ["", ""]+question.utterance.split()+["", ""]
        p = ['', '']+[pp[1] for pp in pos_tagged[i]]+['', '']
        n = ['', '']+[nn[1] for nn in ner_tagged[i]]+['', '']
        l = [4, 4]+labels[i]+[4, 4]

        for j in range(2, len(u)-2):
            features.append(construct_feature(p, u, n, j, l[j-1]))

    return features

def load_questions(fpath):
    """Load a dataset of questions using json

    Keyword arguments:
    fpath -- path to dataset

    """
    return json.load(open(fpath), object_hook=object_decoder)

if __name__ == "__main__":
    sep = os.path.sep
    parser = argparse.ArgumentParser(description="Construct features for phrase detection")
    parser.add_argument("fpath", help="Path to files pos_tagged.pickle,ner_tagged.pickle,json dataset of questions and pickled labels of all words", type=str)
    parser.add_argument("size", help="Number of questions to construct features for", type=int, default=0)
    parser.add_argument("type", help="Operation performed,p = POS tagging,n = NER tagging,i = construct features", type=str, default=0)
    parser.add_argument("mode", help="Training or testing mode, required values: trn or tst", type=str, default=0)
    parser.add_argument("st_ner", help="Path to NER tagger", type=str, default=0)
    parser.add_argument("java_path", help="Path to java executable", type=str, default=0)
    args = parser.parse_args()

    path = args.fpath + "data" + sep
    mode = args.mode
    java_path = args.java_path
    st_ner = args.st_ner

    questions = load_questions(path + "free917." + mode + ".examples.canonicalized.json")
    char = args.type.lower()

    if 'p' in char:
        pickle.dump(pos_tag(questions), open(path + "pos_tagged_" + mode + ".pickle","wb"))

    if 'n' in char:
        pickle.dump(ner_tag(questions, st_ner, java_path), open(path + "ner_tagged_" + mode + ".pickle","wb"))

    if 'i' in char:
        pos_tagged = pickle.load(open(path + "pos_tagged_" + mode + ".pickle"))
        ner_tagged = pickle.load(open(path + "ner_tagged_" + mode + ".pickle"))
        size = args.size
        features = create_features(questions, pos_tagged, ner_tagged, path+"questions_" + mode + "_" + str(size) + ".pickle")
        pickle.dump(features, open(path + "phrase_detect_features_" + mode + "_"+str(size)+"_arr.pickle", "wb"))