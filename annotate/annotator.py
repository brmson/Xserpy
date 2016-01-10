import argparse,json,nltk,os
from nltk.tag import StanfordNERTagger,StanfordPOSTagger

class Question(object):
    def __init__(self, utterance, targetFormula):
        self.utterance = utterance
        self.targetFormula = targetFormula

def object_decoder(obj):
    return Question(obj['utterance'], obj['targetFormula'])

def annotate_questions(questions,start):
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

def ner_tag(questions):
    path = 'C:\\Users\Martin\\PycharmProjects\\xserpy\\stanford-nlp\\'
    st_ner = StanfordNERTagger(path+'classifiers\\english.all.3class.distsim.crf.ser.gz',path+'stanford-ner.jar')
    java_path = "C:\\Program Files\\Java\\jdk1.8.0_65\\bin\\java.exe"
    os.environ['JAVAHOME'] = java_path
    tagged = []
    for q in questions:
        text = nltk.word_tokenize(q.utterance)
        tagged.append(st_ner.tag(text))
    return tagged

if __name__  == "__main__":
    parser = argparse.ArgumentParser(description="Annotate questions with DAGs")
    parser.add_argument("fpath",help="filepath",type=str)
    parser.add_argument("start",help="start",type=int,default=0)
    args = parser.parse_args()
    questions = json.load(open(args.fpath),object_hook=object_decoder)
    # annotate_questions(questions[args.start:],args.start)