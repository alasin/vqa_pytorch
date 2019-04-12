from collections import Counter
import os
import re

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

def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (str(p + ' ') in inText or str(' ' + p) in inText) or (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')	
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    question_file = os.path.join(current_dir, "../data/OpenEnded_mscoco_train2014_questions.json")
    annotation_file = os.path.join(current_dir, "../data/mscoco_val2014_annotations.json")
    vqa_db = VQA(annotation_file, question_file)


    questions = [q['question'] for q in vqa_db.questions['questions']]
    ques_list = []
    for question in questions:
        question = question.lower()[:-1]
        question.replace('?', '') #Just in case
        words = question.split(' ')
        for w in words:
            ques_list.append(w)

    q_vocab = build_vocab(ques_list)
    # print(q_vocab)
    
    answer_objs = [ans_obj['answers'] for ans_obj in vqa_db.dataset['annotations']]
    answers = [a[0]['answer'] for a in answer_objs]
    ans_list = []

    for answer in answers:
        mod_ans = processPunctuation(answer)
        ans_list.append(mod_ans)

    a_vocab = build_vocab(ans_list, k_common=1000)
    # print(a_vocab)
