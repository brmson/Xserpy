import argparse, json, pickle
from phrase_detector import *
from feature_constructor import label_phrases



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
        # print M
        labeled.append(L)
    pickle.dump(labeled, open("questions_train_"+str(start+1)+"_"+str(start+20)+".pickle", "wb"))

def bootstrap(questions, features, labels, step, n_iter, start):
    path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\"
    pos_tagged = pickle.load(open(path + "data\\pos_tagged.pickle"))
    ner_tagged = pickle.load(open(path + "data\\ner_tagged.pickle"))
    examples = zip(features, labels)
    i = start
    L = []
    # ll = open("labels.txt", 'w')
    weights = init_weights(examples, {}, 5)
    while i < len(questions):
        weights = train(n_iter, examples, weights, 5)
        f,  l = label_phrases(questions[i:min(len(questions), i+step)], pos_tagged[i:min(len(questions), i+step)], ner_tagged[i:min(len(questions), i+step)], weights)
        # for L in l:
        #     print L
        #     ll.write(str(L)+"\n")
        L += l
        weights = init_weights(zip(f, l), weights, 5)
        examples = examples + zip(f, l)
        i = i + step
    #     print i
    #     ll.flush()
    # ll.close()
    return labels + L

def parse_to_phrases(questions, labels):
    phrases = []
    for i in range(len(questions)):
        u = questions[i].utterance.split()
        label = labels[i]
        dic = {}
        phrase = ["", "", "", ""]
        order = [0, 0, 0, 0]
        j = 0
        for index in range(len(label)):
            l = label[index]
            word = u[index]
            index += 1
            if l == 4:
                continue
            if l not in dic.keys():
                dic[l] = j
                order[j] = l
                j += 1
            phrase[dic[l]] += word + " "
        phrases.append(zip(phrase, order))
    return phrases

def parse_phrases(questions, labels):
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

def parse_dags(phrases):
    dags = []
    for phrase in phrases:
        dag = [[] for e in range(len(phrase))]
        for i in range(len(phrase)):
            p = phrase[i]
            # if p[1] == 2 or p[1] == [1]:
            #     continue
            for j in range(i+1, len(phrase)):
                q = phrase[j]
                if p[1] == 0:
                    if q[1] == 1:
                        dag[j].append(i)
                    elif q[1] == 3:
                        dag[i].append(j)
                elif p[1] == 1 and q[1] == 0:
                    dag[i].append(j)
                elif p[1] == 2 and q[1] == 3:
                    dag[i].append(j)
                elif p[1] == 3:
                    if q[1] == 2:
                        dag[j].append(i)
                    elif q[1] == 0:
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
    args = parser.parse_args()

    path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\"
    words = pickle.load(open(path+"annotate\\phrase_detect_features_100_arr.pickle"))
    labels = pickle.load(open(path+"data\\labels_trn_100.pickle"))
    questions = json.load(open(args.fpath), object_hook=object_decoder)
    # annotate_questions_label(questions, 120)
    examples = bootstrap(questions, words, labels, 40, 50, 100)
    # phrases = parse_phrases(questions[:100], labels)
    # dags = parse_dags(phrases)
    pickle.dump(examples, open(path+"data\\all_examples.pickle", "wb"))
