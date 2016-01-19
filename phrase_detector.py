import argparse
import annotate.feature_constructor as fc

def generate_candidates(x,y):
    return

def find_argmax(w,candidates):
    return

def train(questions,labels,n_iter,n):
    learning_rate = 1
    l = len(questions)
    j = 0
    w = [0]*n
    while j < n_iter:
        for i in range(l):
            q = questions[i]
            t = labels[i]
            best = find_argmax(w,generate_candidates(q,t))
            w = w + learning_rate*(-best)
        j += 1
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct features for phrase detection")
    parser.add_argument("fpath",help="filepath",type=str)
    parser.add_argument("n_iter",help="iterations",type=int,default=0)
    args = parser.parse_args()
    train(args.n_iter)
    questions = fc.construct_features(fc.load_questions(args.fpath))