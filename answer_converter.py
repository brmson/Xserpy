from annotate.annotator import object_decoder,Question
import json

if __name__ == "__main__":
    path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\"
    questions = json.load(open(path+"data\\free917.train.examples.canonicalized.json"), object_hook=object_decoder)
    answers = [line.strip() for line in open("free917v2.dev.gold")]
    formulas = [line.strip() for line in open("free917v2.dev.mrl")]
    z = zip(formulas,answers)
    gold = dict(zip(formulas,answers))
    with open("free917_trn_answers.txt","w") as f:
        for q in questions:
            if q.targetFormula in gold.keys():
                print(gold[q.targetFormula])
                f.write(gold[q.targetFormula]+"\n")
            else:
                f.write("\n")