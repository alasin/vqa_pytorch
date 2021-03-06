import torch.nn as nn
import torch

class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self, img_feat_size, q_vocab_size, a_vocab_size):
        super().__init__()
        self.q_embedding_size = 1024
        self.q_vocab_size = q_vocab_size
        self.a_vocab_size = a_vocab_size
        self.img_feat_size = img_feat_size

        self.fc_ques = nn.Linear(self.q_vocab_size, self.q_embedding_size, bias=False)
        self.classifier = nn.Linear(self.img_feat_size + self.q_embedding_size, self.a_vocab_size, bias=False)

    def forward(self, image_encoding, question_encoding, question_encoding_oh, question_length):
        image_encoding = image_encoding.view(image_encoding.shape[0], -1)
        q_embedding = self.fc_ques(question_encoding_oh)
        x = torch.cat((image_encoding, q_embedding), dim=-1)
        out = self.classifier(x)

        return out