from annotator import Question,object_decoder
from nltk import StanfordNERTagger
import nltk,os,argparse

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
        tagged.append(nltk.pos_tag(q.utterance.split()))
    return tagged

if __name == "__main__"
    parser = argparse.ArgumentParser(description="Annotate questions with DAGs")
    parser.add_argument("fpath",help="filepath",type=str)
    parser.add_argument("start",help="start",type=int,default=0)
    args = parser.parse_args()
    questions = json.load(open(args.fpath),object_hook=object_decoder)