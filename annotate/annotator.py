import argparse,json,pickle
from phrase_detector import *
from feature_constructor import label_phrases



class Question(object):
    def __init__(self, utterance, targetFormula):
        self.utterance = utterance
        self.targetFormula = targetFormula

def object_decoder(obj):
    return Question(obj['utterance'], obj['targetFormula'])

def annotate_questions_dag(phrases):
    dag = []
    for p in phrases:
        if ("",0) in p:
            p.remove(("",0))
        print p
        d = []
        for token in p:
            edges = raw_input(token).strip().split()
            d.append(edges)
        dag.append(d)
    return dag

def annotate_questions_label(questions):
    labeled = []
    dic = {'e': 0,'r': 1,'c': 2,'v': 3,'n': 4}
    for q in questions[50:70]:
        print q.utterance
        L = []
        l = q.utterance.split()
        for word in l:
            label = raw_input(word+" ")
        # print M
        labeled.append(L)
    pickle.dump(labeled,open("questions_train_51_70.pickle","wb"))

def bootstrap(questions,features,labels,step,n_iter,start):
    path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\"
    pos_tagged = pickle.load(open(path + "data\\pos_tagged.pickle"))
    ner_tagged = pickle.load(open(path + "data\\ner_tagged.pickle"))
    examples = zip(features,labels)
    i = start
    weights = init_weights(examples,{},5)
    while i < len(questions):
        weights = train(n_iter,examples,weights,5)
        f, l = label_phrases(questions[i:min(len(questions),i+step)],pos_tagged[i:min(len(questions),i+step)],ner_tagged[i:min(len(questions),i+step)],weights)
        weights = init_weights(zip(f,l),weights,5)
        examples = examples + zip(f,l)
        i = i + step
    return examples

def parse_to_phrases(questions,labels):
    phrases = []
    print len(questions),len(labels)
    for i in range(len(questions)):
        u = questions[i].utterance.split()
        label = labels[i]
        dic = {}
        phrase = ["","","",""]
        order = [0,0,0,0]
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
        phrases.append(zip(phrase,order))
    return phrases

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate questions with DAGs")
    parser.add_argument("fpath",help="filepath",type=str)
    parser.add_argument("start",help="start",type=int,default=0)
    args = parser.parse_args()

    path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\"
    words = pickle.load(open(path+"annotate\\phrase_detect_features_90_arr.pickle"))
    labels = pickle.load(open(path+"data\\labels_trn_90.pickle"))
    questions = json.load(open(args.fpath),object_hook=object_decoder)
    # annotate_questions_label(questions)
    bootstrap(questions,words,labels,50,10,89)
    # phrases = parse_to_phrases(questions[:20],labels)
    # dags = annotate_questions_dag(phrases)
    # pickle.dump(dags,open("dags_20.pickle","wb"))
