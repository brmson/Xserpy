from annotate.annotator import object_decoder,Question
import json

if __name__ == "__main__":
    path = "C:\\Users\\Martin\\PycharmProjects\\xserpy\\"
    questions = json.load(open(path+"data\\free917.tst.examples.canonicalized.json"), object_hook=object_decoder)
    answers = [line.strip() for line in open("data\\free917v2.test.gold")]
    formulas = [line.strip() for line in open("data\\free917v2.test.mrl")]
    z = zip(formulas,answers)
    gold = dict(zip(formulas,answers))
    with open("data\\free917_tst_answers.txt","w") as f:
        for q in questions:
            if q.targetFormula in gold.keys():
                f.write(gold[q.targetFormula]+"\n")
            else:
                print(q.utterance)
                f.write("\n")