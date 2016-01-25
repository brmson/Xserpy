import os,argparse,json,pickle,nltk,time
from nltk import StanfordNERTagger

from annotator import object_decoder

def uni_bi_tri(l,dictionary,ls,j):
    u = [0]*l
    u[dictionary[ls[j][1]]] = 1

    bl = l**2
    b = bl * 2 * [0]
    b[sub2ind(dictionary[ls[j-1][1]],dictionary[ls[j][1]],0,l)] = 1
    b[sub2ind(dictionary[ls[j][1]],dictionary[ls[j+1][1]],0,l) + bl] = 1

    # tl = l*bl
    # t = tl * 3 * [0]
    # t[sub2ind(dictionary[ls[j-2][1]],dictionary[ls[j-1][1]],dictionary[ls[j][1]],l)] = 1
    # t[sub2ind(dictionary[ls[j-1][1]],dictionary[ls[j][1]],dictionary[ls[j+1][1]],l) + tl] = 1
    # t[sub2ind(dictionary[ls[j][1]],dictionary[ls[j+1][1]],dictionary[ls[j+2][1]],l) + 2*tl] = 1
    return u + b # + t

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

def construct_features(questions):
    path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\data\\"
    tagdict = dict([(item,index) for index,item in list(enumerate(nltk.data.load('help\\tagsets\\upenn_tagset.pickle').keys()))])
    tagdict[''] = 36

    p = len(tagdict.keys())
    pos_tagged = pickle.load(open(path + "pos_tagged.pickle"))

    l = 5
    labels = pickle.load(open(path + "questions_trn_90.pickle"))
    # ner_tagged = ner_tag(questions)

    words = list(enumerate(['','']+list(set([item for sublist in [q.utterance.split() for q in questions] for item in sublist]))+['','']))
    worddict = dict([(item,index) for index,item in words])
    w = len(words)

    # labeled = pickle.load(open("questions_train.pickle"))
    word_index = 2
    features = []

    for i in range(len(questions)):
        question = questions[i]
        pos = [('',''),('','')] + pos_tagged[i] + [('',''),('','')]

        # ner = ner_tagged[i]
        u = question.utterance.split()

        lab = [4,4] + labels[i]

        for j in range(2,len(pos)-2):
            p_f = uni_bi_tri(p,tagdict,pos,j)
            w_f = w*[0]
            print u,pos
            # print u[j-2]
            w_f[worddict[u[j-2]]] = 1  # uni_bi_tri(w,worddict,words,word_index)

            word_index += 1
            # n_f = [0]

            l_f = l*[0]
            l_f[lab[j-1]] = 1

            l_fb = w*l*[0]
            l_fb[lab[j-1]*w+worddict[u[j-2]]] = 1

            features.append(p_f + w_f + l_f + l_fb)
    return features

def load_questions(fpath):
    return json.load(open(fpath),object_hook=object_decoder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct features for phrase detection")
    parser.add_argument("fpath",help="filepath",type=str)
    parser.add_argument("start",help="start",type=int,default=0)
    args = parser.parse_args()
    questions = load_questions(args.fpath)
    c = construct_features(questions[:90])
    pickle.dump(c,open("phrase_detect_features.pickle","wb"))