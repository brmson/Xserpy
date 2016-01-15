from annotator import Question,object_decoder
from nltk import StanfordNERTagger
import nltk,os,argparse,json,pickle

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

def pos_tag(questions):
    tagged = []
    for q in questions:
        text = nltk.word_tokenize(q.utterance)
        tagged.append(nltk.pos_tag(text))
    return tagged

def construct_features(questions):
    tagdict = dict([(item,index) for index,item in list(enumerate(nltk.data.load('help\\tagsets\\upenn_tagset.pickle').keys()))])
    t = len(tagdict.keys())
    pos_tagged = pos_tag(questions)
    ner_tagged = ner_tag(questions)
    words = [item for sublist in [q.utterance.split() for q in questions] for item in sublist]
    labeled = pickle.load(open("question_train.pickle"))
    word_index = 0
    features = []
    for i in range(len(questions)):
        question = questions[i]
        pos = pos_tagged[i]
        ner = ner_tagged[i]
        # u = question.utterance.split()
        for j in range(len(pos)):
            feature = [0]*t
            feature[tagdict[pos[2]]] = 1


def load_questions(fpath):
    return json.load(open(fpath),object_hook=object_decoder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct features for phrase detection")
    parser.add_argument("fpath",help="filepath",type=str)
    parser.add_argument("start",help="start",type=int,default=0)
    args = parser.parse_args()
    questions = load_questions(args.fpath)
    construct_features(questions[:19])