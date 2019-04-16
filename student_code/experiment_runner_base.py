from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

import tensorboardX as tbx

# There must be some cleaner way to do this
from student_code.vqa_dataset import collate_fn

class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 10  # Steps
        self._test_freq = 250  # Steps

        self._train_dataset_loader = DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_data_loader_workers,
                                                pin_memory=True,
                                                collate_fn=collate_fn)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_data_loader_workers,
                                              pin_memory=True,
                                              collate_fn=collate_fn)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()

        if self._cuda:
            self._model = self._model.cuda()

        self.tb_logger = tbx.SummaryWriter()

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self):
        correct_answers = 0
        total_answers = 0

        self._model.eval()
        for batch_id, batch_data in enumerate(self._val_dataset_loader):
            image_encoding = batch_data['image_enc']
            question_encoding = batch_data['ques_enc']
            question_encoding_oh = batch_data['ques_enc_oh']
            question_length = batch_data['ques_len']
            ground_truth_answer = batch_data['ans_enc']
            if self._cuda:
                image_encoding = image_encoding.cuda(async=True)
                question_encoding = question_encoding.cuda(async=True)
                question_encoding_oh = question_encoding_oh.cuda(async=True)
                question_length = question_length.cuda(async=True)
                ground_truth_answer = ground_truth_answer.cuda(async=True)
            
            batch_size = ground_truth_answer.shape[0]

            logits = self._model(image_encoding, question_encoding, question_encoding_oh, question_length)
            probs = F.softmax(logits, dim=-1)
            predicted_answer = torch.argmax(probs, dim=-1)

            counts = ground_truth_answer[torch.arange(batch_size), predicted_answer]
            if self._cuda:
                counts = counts.cpu()
            correct_answers = correct_answers + float(torch.sum(torch.min(counts/3, torch.ones(1))))
            
            total_answers = total_answers + batch_size

        return (correct_answers / total_answers) * 100

    def train(self):
        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)

            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                # ============
                # Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                image_encoding = batch_data['image_enc']
                question_encoding = batch_data['ques_enc']
                question_encoding_oh = batch_data['ques_enc_oh']
                question_length = batch_data['ques_len']
                ground_truth_answer = batch_data['ans_enc']
                if self._cuda:
                    image_encoding = image_encoding.cuda(async=True)
                    question_encoding = question_encoding.cuda(async=True)
                    question_encoding_oh = question_encoding_oh.cuda(async=True)
                    question_length = question_length.cuda(async=True)
                    ground_truth_answer = ground_truth_answer.cuda(async=True)

                # Not really predicted answers but logits
                predicted_answer = self._model(image_encoding, question_encoding, question_encoding_oh, question_length)
                # ============

                # Optimize the model according to the predictions
                loss = self._optimize(predicted_answer, ground_truth_answer)

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))
                    self.tb_logger.add_scalar('train/loss', loss, current_step)

                # if current_step % self._test_freq == 0:
                if batch_id == (num_batches - 1):
                    self._model.eval()
                    val_accuracy = self.validate()
                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    self.tb_logger.add_scalar('val/accuracy', val_accuracy, current_step)