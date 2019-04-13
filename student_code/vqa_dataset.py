from torch.utils.data import Dataset
from external.vqa.vqa import VQA

from os import listdir
from os.path import isfile, join
import os
import numpy as np
from PIL import Image
import json
from collections import Counter
import re
import torch

class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self,
                image_dir,
                question_json_file_path,
                annotation_json_file_path,
                image_filename_pattern,
                img_features_dir,
                vocab_json_filename,
                cache_ds_json=False,
                ds_json_filename='temp.json'):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
            img_features_dir (string): Path to the directory with image features
            ds_json_filename (string): Path to the existing dataset json or where to save
            cache_ds_json (string): Save or not

        """
        if os.path.isfile(ds_json_filename):
            f = open(ds_json_filename, "r")
            self.dataset = json.load(f)
            f.close()
        else:
            vqa_db = VQA(annotation_file=annotation_json_file_path, question_file=question_json_file_path)

            self.max_words_in_ques = -1
            
            self.dataset = []
            self.weight = {'yes':2, 'maybe':1, 'no':0}

            print("Populating data structures...")
            i = 0
            for q_id, annotation in vqa_db.qa.items():
                entry = {}
                question = vqa_db.qqa[q_id]['question']
                question = question.lower()[:-1]
                question = question.replace('?', '') #Just in case
                words = question.split(' ')
                if len(words) > self.max_words_in_ques:
                    self.max_words_in_ques = len(words)
                entry['ques'] = words
                answer_objs = annotation['answers']

                possible_answers = [a['answer'] for a in answer_objs]

                entry['possible_answers'] = []
                for answer in possible_answers:
                    mod_ans = self._handle_punctuation(answer)
                    entry['possible_answers'].append(mod_ans)

                
                img_full_idx = "%012d" % annotation['image_id']
                img_name = image_filename_pattern.replace('{}', img_full_idx)
                img_loc = os.path.join(image_dir, img_name)
                entry['img_loc'] = img_loc

                img_feature_loc = os.path.join(img_features_dir, img_name.replace('.jpg', '.npy'))
                entry['img_feat_loc'] = img_feature_loc
                self.dataset.append(entry)

            if cache_ds_json:
                f = open(ds_json_filename, "w")
                json.dump(self.dataset, f)
                f.close()
        
        f = open(vocab_json_filename, "r")
        vocab = json.load(f)
        f.close()

        self.q_word_vocab = vocab['q']
        self.q_vocab_size = len(self.q_word_vocab.keys())
        
        self.a_vocab = vocab['a']
        self.a_vocab_size = len(self.a_vocab.keys())


    def _get_q_encoding(self, question):
        vec = np.zeros(self.q_vocab_size, dtype=np.float32)
        for token in question:
            if token in self.q_word_vocab:
                vec[self.q_word_vocab[token]] = 1

        vec = torch.from_numpy(vec)

        return vec

    def _get_a_encoding(self, answers):
        vec = np.zeros(self.a_vocab_size, dtype=np.float32)
        for answer in answers:
            if answer in self.a_vocab:
                vec[self.a_vocab[answer]] += 1

        vec = torch.from_numpy(vec)

        return vec

    def _handle_punctuation(self, inText):
        periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
        commaStrip   = re.compile("(\d)(\,)(\d)")
        punct        = [';', r"/", '[', ']', '"', '{', '}',
                        '(', ')', '=', '+', '\\', '_', '-',
                        '>', '<', '@', '`', ',', '?', '!']
        outText = inText
        for p in punct:
            if (str(p + ' ') in inText or str(' ' + p) in inText) or (re.search(commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')	
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    def _get_answer_class(self, answer):
        if answer in self.a_vocab:
            return self.a_vocab[answer]
        return 0    #Defaulting to 1st answer
    

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        entry = self.dataset[idx]
        img_feat = np.load(entry['img_feat_loc'])
        ques = entry['ques']
        ques_enc = self._get_q_encoding(ques)

        possible_answers = entry['possible_answers']
        ans_encoding = self._get_a_encoding(possible_answers)

        return {'image_enc': img_feat, 'ques_enc': ques_enc, 'ans_enc': ans_encoding}