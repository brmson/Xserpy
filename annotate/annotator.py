import json, os.path

from phrase_detection.phrase_detector import *
from phrase_detection.feature_constructor import label_phrases


class Question(object):
    def __init__(self,  utterance,  targetFormula):
        self.utterance = utterance
        self.targetFormula = targetFormula

def object_decoder(obj):
    return Question(obj['utterance'],  obj['targetFormula'])

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
    labeled = []
    dic = {'e': 0, 'r': 1, 'c': 2, 'v': 3, 'n': 4}
    for q in questions[start:start+20]:
        print q.utterance
        L = []
        l = q.utterance.split()
        for word in l:
            inp = raw_input(word+" ")
            while inp not in dic.keys():
                inp = raw_input("oprava: ")
            label = dic[inp]
            L.append(label)
        labeled.append(L)
    pickle.dump(labeled, open("questions_test_"+str(start+1)+"_"+str(start+20)+".pickle", "wb"))

def bootstrap(questions, features, labels, step, n_iter, start):
    path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\"
    pos_tagged = pickle.load(open(path + "data\\pos_tagged.pickle"))
    ner_tagged = pickle.load(open(path + "data\\ner_tagged.pickle"))
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
    phrases = []
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
                while label[j] == 4:
                    j += 1
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
        phrases.append(zip(phrase, order))
    return phrases

def count_entities(phrase):
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

    if 'b' in type:
        step = 40
        n_iter = 50
        examples = bootstrap(questions, words, labels, step, n_iter, start)
        pickle.dump(examples, open(path+"data\\b_examples" + mode + "_" + str(size) + ".pickle", "wb"))
    elif 'l' in type:
        annotate_questions_label(questions, start)
    else:
        labels = pickle.load(open(path+"data" + sep + "questions_" + mode + "_" + str(size) + ".pickle"))
        dags = annotate_questions_dag(parse_to_phrases(questions, labels))
        pickle.dump(dags, open(path+"data" + sep + "dag_m_examples_" + mode + "_" + str(size) + ".pickle", "wb"))
