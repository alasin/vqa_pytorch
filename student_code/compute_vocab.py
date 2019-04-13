from collections import Counter
import os
import re
import json

import sys
sys.path.append('../')
from external.vqa.vqa import VQA

periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip   = re.compile("(\d)(\,)(\d)")
punct        = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']

def build_vocab(data, k_common=None):
    counter = Counter(data)
    words = counter.keys()
    if k_common:
        words = counter.most_common(n=k_common)
        words = [word for word, count in words]

    tokens = sorted(words, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens)}
    return vocab

def handle_punctuation(inText):
    outText = inText
    for p in punct:
        if (str(p + ' ') in inText or str(' ' + p) in inText) or (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')	
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText


if __name__ == "__main__":

    question_file = sys.argv[1]
    annotation_file = sys.argv[2]
    out_filename = sys.argv[3]
    vqa_db = VQA(annotation_file, question_file)

    ques_list = []
    ans_list = []
    for q_id, annotation in vqa_db.qa.items():
        question = vqa_db.qqa[q_id]['question']
        question = question.lower()[:-1]
        question = question.replace('?', '') #Just in case
        words = question.split(' ')
        ques_list += words

        answer_objs = annotation['answers']

        possible_answers = [a['answer'] for a in answer_objs]

        for answer in possible_answers:
            mod_ans = handle_punctuation(answer)
            ans_list.append(mod_ans)

    q_vocab = build_vocab(ques_list)
    a_vocab = build_vocab(ans_list, k_common=2000)
    vocabs = {'q': q_vocab, 'a': a_vocab}

    f = open(out_filename, "w")
    json.dump(vocabs, f)
    f.close()
