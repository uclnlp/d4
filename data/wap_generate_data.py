import json
from util.corenlp.corenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000/')


with open("./data/wap/questions.json") as json_file:
    data = json.load(json_file)

new_data = {}

# 'iIndex', 'sQuestion', 'lAlignments', 'lEquations', 'lSolutions'
i = 0
for elem in data:
    i += 1
    text = elem['sQuestion']
    output = nlp.annotate(text,
                          properties={'annotators': 'tokenize,ssplit',
                                      'outputFormat': 'json'}
                          )

    tokens = [token['word'] for sentence in output['sentences'] for token in sentence['tokens']]

    new_data[elem['iIndex']] = (elem['lSolutions'][0], " ".join(tokens))

for dataset in ['train', 'test', 'dev']:
    with open("./data/wap/{}.tsv".format(dataset), "r") as f_in, \
         open("./data/wap/{}.txt".format(dataset), "w") as f_out:
        for line in f_in:
            id_ = line.rstrip()
            f_out.write("{0}\t{1}\n".format(new_data[int(id_)][0], new_data[int(id_)][1]))
