from student_code.coattention_net import CoattentionNet, QuestionProcessor
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset

import torch

class CoattentionNetExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Co-Attention model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 train_img_feat_path, test_image_dir, test_question_path, test_annotation_path,
                 test_img_feat_path, vocab_path, batch_size, num_epochs,
                 num_data_loader_workers):

        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg",
                                   img_features_dir=train_img_feat_path,
                                   vocab_json_filename=vocab_path)
        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg",
                                 img_features_dir=test_img_feat_path,
                                 vocab_json_filename=vocab_path)

        img_feat_size = 512 #TODO Better way to do this
        embedding_size = 512

        q_vocab_size = train_dataset.q_vocab_size
        a_vocab_size = train_dataset.a_vocab_size
        self._model = CoattentionNet(img_feat_size, embedding_size, q_vocab_size, a_vocab_size).cuda()

        params = self._model.parameters()

        # self.optimizer = torch.optim.RMSprop(params=params, lr=4e-4, weight_decay=1e-8, momentum=0.99)
        self.optimizer = torch.optim.Adam(params=params, lr=1e-4)

        self.criterion = torch.nn.CrossEntropyLoss().cuda()

        super().__init__(train_dataset, val_dataset, self._model, batch_size, num_epochs,
                         num_data_loader_workers=num_data_loader_workers)

    def _optimize(self, predicted_answers, true_answer_ids):
        majority_ans = torch.argmax(true_answer_ids, dim=-1)
        loss = self.criterion(predicted_answers, majority_ans)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
