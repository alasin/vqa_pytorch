from student_code.simple_baseline_net import SimpleBaselineNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset
import torch

class SimpleBaselineExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Simple Baseline model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers):

        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg",
                                   img_features_dir='features/img_train')
        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg",
                                 img_features_dir='features/img_val')

        img_feat_size = 1024 #TODO Better way to do this
        ques_embedding_lr = 0.8
        classifier_lr = 0.01

        q_vocab_size = train_dataset.q_vocab_size
        a_vocab_size = train_dataset.a_vocab_size
        model = SimpleBaselineNet(img_feat_size, q_vocab_size, a_vocab_size)

        
        self.optimizer = torch.optim.SGD([{'params': model.fc_ques.parameters(), 'lr': ques_embedding_lr},
                                          {'params': model.classifier.parameters(), 'lr': classifier_lr}])

        self.criterion = torch.nn.CrossEntropyLoss()

        super().__init__(train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers)

    def _optimize(self, predicted_answers, true_answer_ids):
        loss = self.criterion(predicted_answers, true_answer_ids)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.model.fc_ques.weight.data.clamp_(-1500, 1500)
        self.model.classifier.weight.data.clamp_(20, 20)
