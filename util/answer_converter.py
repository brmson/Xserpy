"""Add gaps for missing answers so that one line equals one answer for each question"""
from annotate.annotator import object_decoder,Question
import json, os, sys

if __name__ == "__main__":
    sep = os.path.sep
    path = sys.argv[1]
    mode = sys.argv[2]
    questions = json.load(open(path + "data" + sep + "free917." + mode + ".examples.canonicalized.json"), object_hook=object_decoder)
    answers = [line.strip() for line in open(path + "data" + sep + "free917v2." + mode + ".gold")]
    formulas = [line.strip() for line in open(path + "data" + sep + "free917v2." + mode + ".mrl")]
    z = zip(formulas,answers)
    gold = dict(zip(formulas,answers))
    with open(path + "data" + sep + "free917_" + mode + "_answers.txt","w") as f:
        for q in questions:
            if q.targetFormula in gold.keys():
                f.write(gold[q.targetFormula]+"\n")
            else:
                print(q.utterance)
                f.write("\n")