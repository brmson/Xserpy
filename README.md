# Xserpy

Python implementation of Shift-Reduce semantic parser: http://ceur-ws.org/Vol-1180/CLEF2014wn-QA-XuEt2014.pdf

## Instructions for running
Run each script with -h parameter to see a list of required parameters. These usually consist of path to data files, training/testing mode(trn/tst), size of dataset and operating mode. For scripts where training a model is involved, number of iterations is usually used. Description of what each operating mode of each script does follows. Words in brackets in file names are variable parameters. Files not mentioned have main methods only for testing their methods which other scripts import. All scripts require Free917 dataset, training and testing split.
**Annotate**
**annotator**
* l – annotate all words in dataset with phrase detection labels; creates ``labels_(trn/tst)_(size).pickle`` and ``questions_(trn/tst)_(size).pickle`` files.
* b – labels all words in dataset using bootstraping; creates ``labels_(trn/tst)_(size).pickle`` and ``questions_(trn/tst)_(size).pickle`` files
**Phrase_detection**
**feature_constructor**
* p – tags all questions in dataset with part-of-speech tags; creates ``pos_tagged_(trn/tst).pickle`` file
* n – tags all questions in dataset with NER tags; creates ``ner_tagged_(trn/tst).pickle`` file
* i – creates features for phrase detection for all questions in the dataset. Requires ``pos_tagged_*.pickle``, ``ner_tagged_*.pickle`` and ``questions_(trn/tst)_(size).pickle`` files; creates ``phrase_detect_features_(trn/tst)_(size)_arr.pickle`` file
**feature_constructor**
* l – creates training examples for phrase detection model training. Requires ``labels_(trn/tst)_(size).pickle`` and ``phrase_detect_features_(trn/tst)_(size).pickle`` files; creates ``phr_detect_examples_(trn/tst)_(size).pickle`` and ``empty_weights_(trn/tst)_(size).pickle`` files.
* t – trains phrase detection model. Requires ``phr_detect_examples_(trn/tst)_(size).pickle`` and ``empty_weights_(trn/tst)_(size).pickle`` files; creates ``w_(size)_(iterations).pickle`` file
* e – computes error of a model on a testing set. Requires ``w_641_(iterations).pickle``, ``labels_tst_276.pickle``, ``questions_tst_276.pickle`` and ``phrase_detect_features_tst_276_arr.pickle`` files.
**Shift_reduce**
**shift_reduce** – requires ``labels_(trn/tst)_(size).pickle``, ``questions_(trn/tst)_(size).pickle`` and ``pos_tagged_(trn/tst).pickle`` files
* c – creates training examples for shift-reduce model training. Requires ``gold_dags_(trn/tst)_(size).pickle`` file; ``creates dag_examples_(trn/tst)_(size).pickle``, ``gold_sequences_(trn/tst)_(size).pickle`` and ``empty_weights_dag_(trn/tst)_(size).pickle`` files.
* t – trains shift-reduce model. Requires ``dag_examples_(trn/tst)_(size).pickle`` and ``empty_weights_dag_(trn/tst)_(size).pickle`` files; creates ``w_dag(size)_(iterations).pickle`` file
* b – computes error of a model on a testing set. Requires ``w_dag641_(iterations).pickle``, ``gold_dags_tst_276.pickle`` and ``gold_sequences_tst_276.pickle`` files.
**Query_intention**
**entity _linking** – requires ``questions_(trn/tst)_(size).pickle``
* e – obtain candidates for entity linking through Google Freebase API; creates ``candidates_(trn/tst)_(size).pickle`` file
* g – obtain correct entities for linking. Requires ``candidates_(trn/tst)_(size).pickle`` and ``query_gold_ent_(trn/tst).pickle`` files; creates ``gold_entities_(trn/tst)_(size).pickle`` file
* f – construct features for entity linking. Requires ``gold_entities_(trn/tst)_(size).pickle`` and ``candidates_(trn/tst)_(size).pickle`` files; creates ``candidates_features_(trn/tst)_(size).pickle`` and ``ent_labels_(trn/tst)_(size).pickle`` files
* t – train model for entity linking. Requires ``candidates_features_(trn+tst)_(size).pickle`` and ``ent_labels_(trn+tst)_(size).pickle`` files (4 total); creates ``ent_lr_trn_641.pickle`` file
* r – construct features for relation linking and train model. Requires ``query_gold_rel_trn.pickle`` file; creates ``relation_lr_trn_641.pickle`` file
* u – evaluate model for relation linking. Requires ``relation_lr_trn_641.pickle``,  ``query_gold_rel_tst.pickle`` and ``rel_dict.pickle`` files
* l – construct features for edge linking. Requires ``query_gold_edges_(trn/tst).pickle`` and ``gold_dags_(trn+tst)_(size).pickle`` files. Creates ``edge_features_(trn/tst).pickle`` and  ``edge_labels_(trn/tst).pickle`` files
* d – train model for edge linking. Requires ``edge_features_(trn+tst).pickle`` and  ``edge_labels_(trn+tst).pickle`` files (4 total); ``creates edge_lr_trn.pickle`` file
* a – link all questions to KB. Requires all linking models and ``candidates_(trn/tst)_(size).pickle`` file.
* q – parse logic formulas to linked DAGs; creates  ``query_gold_rel_(trn/tst).pickle``, ``query_gold_ent_(trn/tst).pickle``,  ``query_gold_dags_(trn/tst).pickle`` and  ``query_gold_edges_(trn/tst).pickle`` files
* c – create vocabularies for edges and relations
**Question_conversion**
**convert_question** – requires all models, dictionaries, ``free917_(trn/tst)_answers.txt`` and ``pos_tagged_*.pickle`` files
* i – answers questions input by user
* f – answers questions from file
* a – answers questions from dataset and evaluates them on gold standard answers


 ## Credits
  * **https://github.com/pks/rebol** - for answers for Free917 questions