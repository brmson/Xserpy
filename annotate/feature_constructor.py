import os, argparse, json, pickle, nltk, time
from nltk import StanfordNERTagger
from phrase_detector import predict

class Question(object):
    def __init__(self,  utterance,  targetFormula):
        self.utterance = utterance
        self.targetFormula = targetFormula

def object_decoder(obj):
    return Question(obj['utterance'],  obj['targetFormula'])

def uni_bi_tri(c, u, j):
    feature = []
    feature.append(c+"_u_"+u[j])
    feature.append(c+"_b0_"+u[j-1]+"_"+u[j])
    feature.append(c+"_b1_"+u[j]+"_"+u[j+1])
    feature.append(c+"_t0_"+u[j-2]+"_"+u[j-1]+"_"+u[j])
    feature.append(c+"_t1_"+u[j-1]+"_"+u[j]+"_"+u[j+1])
    feature.append(c+"_t2_"+u[j]+"_"+u[j+1]+"_"+u[j+2])
    return feature

def sub2ind(x, y, z, l):
    return x + l * (y + l * z)

def ner_tag(questions):
    path = 'C:\\Users\Martin\\PycharmProjects\\xserpy\\stanford-nlp\\'
    st_ner = StanfordNERTagger(path+'classifiers\\english.all.3class.distsim.crf.ser.gz', path+'stanford-ner.jar')
    java_path = "C:\\Program Files\\Java\\jdk1.8.0_65\\bin\\java.exe"
    os.environ['JAVAHOME'] = java_path
    tagged = []
    i = 0
    for q in questions:
        text = nltk.word_tokenize(q.utterance)
        tagged.append(st_ner.tag(text))
        i += 1
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

def construct_feature(p, u, n, j, l):
    w_f = uni_bi_tri('w', u, j)
    p_f = uni_bi_tri('p', p, j)
    n_f = uni_bi_tri('n', n, j)

    feature = w_f + p_f + n_f
    feature.append("l_"+str(l))
    feature.append("l_w_"+str(l)+"_"+u[j])
    return feature

def label_phrases(questions, pos_tagged, ner_tagged, weights):
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
    return (features, labels)

def create_features(questions, pos_tagged, ner_tagged, path):

    labels = pickle.load(open(path))

    features = []
    for i in range(len(questions)):
        question = questions[i]
        u = ["", ""]+question.utterance.split()+["", ""]
        p = ['', '']+[pp[1] for pp in pos_tagged[i]]+['', '']
        n = ['', '']+[nn[1] for nn in ner_tagged[i]]+['', '']
        l = [4, 4]+labels[i]+[4, 4]
        for j in range(2, len(u)-2):
            features.append(construct_feature(p, u, n, j, l[j-1]))
    return features

def load_questions(fpath):
    return json.load(open(fpath), object_hook=object_decoder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct features for phrase detection")
    parser.add_argument("fpath", help="Path to files pos_tagged.pickle,ner_tagged.pickle,json dataset of questions and pickled labels of all words", type=str)
    parser.add_argument("size", help="Number of questions to construct features for", type=int, default=0)
    parser.add_argument("type", help="Operation performed,p = POS tagging,n = NER tagging,i = construct features", type=str, default=0)
    parser.add_argument("mode", help="Training or testing mode, required values: trn or tst", type=str, default=0)
    args = parser.parse_args()

    path = args.fpath + "data\\"
    mode = args.mode

    questions = load_questions(args.fpath + "free917." + mode + ".examples.canonicalized.json")
    char = args.type.lower()

    if 'p' in char:
        pickle.dump(pos_tag(questions),open("pos_tagged_" + mode + ".pickle","wb"))

    if 'n' in char:
        pickle.dump(ner_tag(questions),open("ner_tagged_" + mode + ".pickle","wb"))

    if 'i' in char:
        pos_tagged = pickle.load(open(path + "pos_tagged_" + mode + ".pickle"))
        ner_tagged = pickle.load(open(path + "ner_tagged_" + mode + ".pickle"))
        size = args.size
        c = create_features(questions[:size], pos_tagged, ner_tagged, path+"questions_" + mode + "_" + str(size) + ".pickle")
        pickle.dump(c, open("phrase_detect_features_" + mode + "_"+str(size)+"_arr.pickle", "wb"))