import os,argparse,json,pickle,nltk,time
from nltk import StanfordNERTagger
from phrase_detector import predict

class Question(object):
    def __init__(self, utterance, targetFormula):
        self.utterance = utterance
        self.targetFormula = targetFormula

def object_decoder(obj):
    return Question(obj['utterance'], obj['targetFormula'])

def uni_bi_tri(c,u,j):
    feature = []
    feature.append(c+"_u_"+u[j])
    feature.append(c+"_b_"+u[j-1]+"_"+u[j])
    feature.append(c+"_b_"+u[j]+"_"+u[j+1])
    feature.append(c+"_t_"+u[j-2]+"_"+u[j-1]+"_"+u[j])
    feature.append(c+"_t_"+u[j-1]+"_"+u[j]+"_"+u[j+1])
    feature.append(c+"_t_"+u[j]+"_"+u[j+1]+"_"+u[j+2])
    return feature

def sub2ind(x,y,z,l):
    return x + l * (y + l * z)

def ner_tag(questions):
    path = 'C:\\Users\Martin\\PycharmProjects\\xserpy\\stanford-nlp\\'
    st_ner = StanfordNERTagger(path+'classifiers\\english.all.3class.distsim.crf.ser.gz',path+'stanford-ner.jar')
    java_path = "C:\\Program Files\\Java\\jdk1.8.0_65\\bin\\java.exe"
    os.environ['JAVAHOME'] = java_path
    tagged = []
    i = 0
    for q in questions:
        print i
        start = time.time()
        text = nltk.word_tokenize(q.utterance)
        tagged.append(st_ner.tag(text))
        i += 1
        print(time.time()-start)
    return tagged

def pos_tag(questions):
    tagged = []
    i = 0
    for q in questions:
        text = nltk.word_tokenize(q.utterance)
        tagged.append(nltk.pos_tag(text))
        print i
        i += 1
    return tagged

def construct_feature(p,u,n,j,l):
    w_f = uni_bi_tri('w',u,j)
    p_f = uni_bi_tri('p',p,j)
    n_f = uni_bi_tri('n',n,j)

    feature = w_f + p_f + n_f
    feature.append("l_"+str(l))
    feature.append("l_w_"+str(l)+"_"+u[j])
    return feature

def label_phrases(questions,pos_tagged,ner_tagged,weights):
    features = []
    labels = []
    for i in range(len(questions)):
        l = 4
        question = questions[i]
        # print question.utterance
        u = ["",""]+question.utterance.split()+["",""]
        p = ['','']+[pp[1] for pp in pos_tagged[i]]+['','']
        n = ['','']+[nn[1] for nn in ner_tagged[i]]+['','']
        for j in range(2,len(u)-2):
            f = construct_feature(p,u,n,j,l)
            l = predict(weights,f,5)
            features.append(f)
            labels.append(l)
            # print u[j],l,
        # print '\n'
    return (features,labels)

def create_features(questions,pos_tagged,ner_tagged,path):

    labels = pickle.load(open(path))

    features = []
    for i in range(len(questions)):
        question = questions[i]
        u = ["",""]+question.utterance.split()+["",""]
        p = ['','']+[pp[1] for pp in pos_tagged[i]]+['','']
        n = ['','']+[nn[1] for nn in ner_tagged[i]]+['','']
        l = [4,4]+labels[i]+[4,4]
        for j in range(2,len(u)-2):
            features.append(construct_feature(p,u,n,j,l[j-1]))
    return features

def load_questions(fpath):
    return json.load(open(fpath),object_hook=object_decoder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct features for phrase detection")
    parser.add_argument("fpath",help="filepath",type=str)
    parser.add_argument("start",help="start",type=int,default=0)
    args = parser.parse_args()
    questions = load_questions(args.fpath)
    path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\"
    pos_tagged = pickle.load(open(path + "data\\pos_tagged.pickle"))
    ner_tagged = pickle.load(open(path + "data\\ner_tagged.pickle"))
    size = 40
    # weights = pickle.load(open(path+"models\\w_90_50.pickle"))
    c = create_features(questions[:size],pos_tagged,ner_tagged,path+"data\\questions_trn_"+str(size)+".pickle")
    # labelled = label_phrases(questions[90:180],pos_tagged,ner_tagged,weights)
    pickle.dump(c,open("phrase_detect_features_40_arr.pickle","wb"))