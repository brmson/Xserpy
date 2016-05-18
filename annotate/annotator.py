"""Annotate words with phrase labels, bootstrap"""
import json, os.path
from phrase_detection.phrase_detector import *
from phrase_detection.feature_constructor import label_phrases, Question, object_decoder


def annotate_questions_dag(phrases):
    dag = []
    for p in phrases:
        if ("", 0) in p:
            p.remove(("", 0))
        print p
        d = []
        for token in p:
            edges = raw_input(token).strip().split()
            d.append(edges)
        dag.append(d)
    return dag

def annotate_questions_label(questions, start):
    """Label words with phrase labels input by user

    Keyword arguments:
    questions -- list of Question objects
    start -- index where to start

    """
    labeled = []
    i = 0
    dic = {'e': 0, 'r': 1, 'c': 2, 'v': 3, 'n': 4}
    for q in questions[start:]:
        print q.utterance
        L = []
        l = q.utterance.split()
        for word in l:
            inp = raw_input(word+" ")

            while inp not in dic.keys():
                # User made a typo
                inp = raw_input("correction: ")
            label = dic[inp]
            L.append(label)

        labeled.append(L)
        i += 1

        # Regular saving of partial result
        if i % 10 == 0:
            pickle.dump(labeled, open("questions_test_partial.pickle", "wb"))
            print i
    pickle.dump(labeled, open("questions_test_whole.pickle", "wb"))

def bootstrap(questions, features, labels, step, n_iter, start, path):
    """Train a model on already labeled subset, label small part of unlabeled subset, train new model, repeat

    Keyword arguments:
    questions -- list of Question objects
    features -- list of features of words
    labels -- list of labels of words
    step -- how many words should be added to the training subset
    n_iter -- number of iterations for model training
    start -- index where unlabeled set starts
    path -- path to pickle files

    """
    sep = os.path.sep
    pos_tagged = pickle.load(open(path + "data" + sep + "pos_tagged.pickle"))
    ner_tagged = pickle.load(open(path + "data" + sep + "ner_tagged.pickle"))
    examples = zip(features, labels)
    i = start
    L = []
    weights = init_weights(examples, {}, 5)
    while i < len(questions):
        weights = train(n_iter, examples, weights, 5)
        f,  l = label_phrases(questions[i:min(len(questions), i+step)], pos_tagged[i:min(len(questions), i+step)], ner_tagged[i:min(len(questions), i+step)], weights)
        L += l
        weights = init_weights(zip(f, l), weights, 5)
        examples = examples + zip(f, l)
        i = i + step
    return labels + L

def parse_to_phrases(questions, labels):
    """Group words into phrases according to their labels

    Keyword arguments:
    questions -- list of Question objects
    labels -- list of lists of phrase labels

    """
    phrases = []

    # Similar to the method in shift_reduce.shift_reduce
    for i in range(len(questions)):
        u = [q for q in questions[i].utterance.split()]
        label = [l for l in labels[i]]
        j = 0
        while label[j] == 4:
            j += 1
        phrase = [u[j]+" "]
        k = 0
        order = [label[j]]
        while j < len(label):
            if j + 1 >= len(label):
                break
            if label[j+1] == label[j]:
                phrase[k] += u[j+1] + " "
                u.remove(u[j+1])
                label.remove(label[j+1])
            else:
                j += 1
                k += 1
                over = False
                while label[j] == 4:
                    j += 1
                    if j == len(label):
                        over = True
                        break
                if not over:
                    phrase.append(u[j] + " ")
                    order.append(label[j])
        m = 0
        while m < len(phrase):
            if order[m] == 0:
                m += 1
                continue
            n = m + 1
            while n < len(phrase):
                if order[m] == order[n]:
                    phrase[m] += phrase[n]
                    del phrase[n]
                    del order[n]
                n += 1
            m += 1
        z = zip(phrase, order)
        zvar = [zz for zz in z if zz[1] == 3]
        zrel = [zz for zz in z if zz[1] == 1]
        zent = [zz for zz in z if zz[1] == 0]
        phrases.append(zvar + zrel + zent)
    return phrases

def count_entities(phrase):
    """Count number of 'entity' phrases in a question

    Keyword arguments:
    phrase -- words of one question grouped into phrases with type included

    """
    result = 0
    index = 0
    for i in range(len(phrase)):
        if phrase[i][1] == 0:
            result += 1
        elif phrase[i][1] == 1:
            index = i
    return result, index

def parse_dags(phrases):
    """ phrases is a list of phrase variables, where each phrase
    is a list of tuples of (string, number) where number is entity type """
    dags = []
    for phrase in phrases:
        result, index = count_entities(phrase)
        dag = [[] for e in range(len(phrase))]
        if result > 1:
            e = range(len(phrase))
            e.remove(index)
            for E in e:
                dag[index].append(E)
        else:
            for i in range(len(phrase)):
                p = phrase[i]
                for j in range(i+1, len(phrase)):
                    q = phrase[j]

                    if p[1] == 0:
                        if q[1] == 1:
                            dag[j].append(i)
                        # elif q[1] == 3:
                        #     dag[i].append(j)

                    elif p[1] == 1:
                        if q[1] == 0:
                            dag[i].append(j)
                        elif q[1] == 3:
                            dag[i].append(j)

                    elif p[1] == 2 and q[1] == 3:
                        dag[i].append(j)

                    elif p[1] == 3:
                        if q[1] == 2:
                            dag[j].append(i)
                        elif q[1] == 1:
                            dag[j].append(i)
        dags.append(dag)
    return dags

def examples_to_phrases(examples, questions):
    i = 0
    result = []
    for q in questions:
        phrase = []
        u = q.utterance.split()
        for U in u:
            phrase.append(examples[i])
            i += 1
        result.append(phrase)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate questions with DAGs")
    parser.add_argument("fpath", help="filepath", type=str)
    parser.add_argument("start", help="start", type=int, default=0)
    parser.add_argument("size", help="Dataset size", type=int, default=0)
    parser.add_argument("type", help="Operating mode", type=str)
    parser.add_argument("mode", help="Training/testing", type=str)
    args = parser.parse_args()
    path = args.fpath
    mode = args.mode
    type = args.type
    size = args.size
    start = args.start
    sep = os.path.sep

    words = pickle.load(open(path+"data" + sep + "phrase_detect_features_" + mode + "_" + str(size) + "_arr.pickle"))
    labels = pickle.load(open(path+"data" + sep + "labels_" + mode + "_" + str(size) + ".pickle"))
    questions = json.load(open(args.fpath + "data" + sep + "free917." + mode + ".examples.canonicalized.json"), object_hook=object_decoder)

    # Mode for bootstraping
    if 'b' in type:
        step = 40
        n_iter = 50
        examples = bootstrap(questions, words, labels, step, n_iter, start, path)
        pickle.dump(examples, open(path+"data\\b_examples" + mode + "_" + str(size) + ".pickle", "wb"))

    # Mode for labelling words with phrase labels
    elif 'l' in type:
        annotate_questions_label(questions, start)
