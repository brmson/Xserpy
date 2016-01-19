import argparse,json,pickle

class Question(object):
    def __init__(self, utterance, targetFormula):
        self.utterance = utterance
        self.targetFormula = targetFormula

def object_decoder(obj):
    return Question(obj['utterance'], obj['targetFormula'])

def annotate_questions_dag(questions,start):
    i = start
    for q in questions:
        M = []
        print i
        # print range(len(q.utterance.split()))
        l = q.utterance.split()
        print list(enumerate(l))
        for word in l:
            edges = raw_input(word+" ")
            M.append([int(e) for e in edges.split()])
        print M
        i += 1


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate questions with DAGs")
    parser.add_argument("fpath",help="filepath",type=str)
    parser.add_argument("start",help="start",type=int,default=0)
    args = parser.parse_args()

    questions = json.load(open(args.fpath),object_hook=object_decoder)
    annotate_questions_label(questions)