import torch.nn as nn
import torch


class ImageProcessor(nn.Module):
    def __init__(self, init_image_embedding_size, embedding_size):
        super().__init__()
        self.conv = nn.Conv2d(init_image_embedding_size, embedding_size, kernel_size=1)

    def forward(self, image_encoding):
        x = self.conv(image_encoding)
        x_size = x.size()
        x = x.view(x_size[0], x_size[1], -1)
        # print(x.shape)
        return x

class QuestionProcessor(nn.Module):
    def __init__(self, q_vocab_size, embedding_size):
        super().__init__()
        self.word_embedding = nn.Embedding(q_vocab_size, embedding_size, padding_idx=0)

        self.phrase_unigram = nn.Conv1d(embedding_size, embedding_size, kernel_size=1, stride=1, padding=0)
        self.phrase_bigram = nn.Conv1d(embedding_size, embedding_size, kernel_size=2, stride=1, padding=1)
        self.phrase_trigram = nn.Conv1d(embedding_size, embedding_size, kernel_size=3, stride=1, padding=1)
        
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.Tanh()

        # Original code uses num_layers=2
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=embedding_size, num_layers=1, batch_first=True)

    def forward(self, question_encoding, question_length):
        w_embed = self.word_embedding(question_encoding)
        
        w_embed_t = torch.transpose(w_embed, 1, 2)
        uni_embed = self.phrase_unigram(w_embed_t)
        bi_embed = self.phrase_bigram(w_embed_t)[:, :, :-1]
        tri_embed = self.phrase_trigram(w_embed_t)
        p_embed = torch.stack([uni_embed, bi_embed, tri_embed], dim=-1)
        p_embed = torch.transpose(p_embed, 1, 2)
        p_embed, _ = torch.max(p_embed, dim=-1)
        p_embed = self.activation(p_embed)
        p_embed = self.dropout(p_embed) #Shape - (batch*max_ques_length*512)

        total_length = p_embed.size(1) #Needed to unpack

        packed = nn.utils.rnn.pack_padded_sequence(p_embed, question_length, batch_first=True)
        q_embed, (_, _) = self.lstm(packed)
        q_embed, input_sizes = nn.utils.rnn.pad_packed_sequence(q_embed, batch_first=True, total_length=total_length)

        # print(w_embed.shape, p_embed.shape, q_embed.shape)
        return (w_embed, p_embed, q_embed)



class Attention(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size    
        
        self.ques_linear = nn.Linear(self.embedding_size, self.hidden_size)
        self.img_linear = nn.Linear(self.embedding_size, self.hidden_size)
        
        self.ques_linear_t = nn.Linear(self.hidden_size, 1)
        self.img_linear_t = nn.Linear(self.hidden_size, 1)

        self.affinity = nn.Linear(self.embedding_size, self.hidden_size, bias=False)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, ques_embed_t, img_embed):
        # Assuming <batch_size x max length x 512>
        # img_feat_size = img_embed_full.size()
        # img_embed = img_embed_full.view(img_feat_size[0], img_feat_size[1], -1)
        ques_embed = torch.transpose(ques_embed_t, 1, 2)
        img_embed_t = torch.transpose(img_embed, 1, 2)
        C = torch.matmul(self.affinity(ques_embed_t), img_embed)
        C = self.activation(C)
        C_t = torch.transpose(C, 1, 2)

        a = torch.transpose(self.img_linear(img_embed_t), 1, 2)
        b = torch.transpose(self.ques_linear(ques_embed_t), 1, 2)
        h_vis = a + torch.matmul(b, C)
        h_vis = self.activation(h_vis)

        a = torch.transpose(self.img_linear(img_embed_t), 1, 2)
        b = torch.transpose(self.ques_linear(ques_embed_t), 1, 2)
        h_ques = b + torch.matmul(a, C_t)
        h_ques = self.activation(h_ques)

        attention_vis = torch.squeeze(self.img_linear_t(torch.transpose(h_vis, 1, 2)), -1)
        attention_ques = torch.squeeze(self.ques_linear_t(torch.transpose(h_ques, 1, 2)), -1)
        attention_vis = self.softmax(attention_vis)
        attention_ques = self.softmax(attention_ques)
        # print(attention_ques.shape)
        # print(attention_vis.shape)
        
        # attention_vis = attention_vis.view(img_feat_size[0], img_feat_size[2], img_feat_size[3])
        attention_vis = attention_vis.unsqueeze(1)
        attention_ques = attention_ques.unsqueeze(1)

        attention_feat_vis = torch.mul(img_embed, attention_vis)
        attention_feat_vis = torch.sum(attention_feat_vis, dim=-1)
        # print(attention_feat_vis.shape)
        attention_feat_ques = torch.mul(ques_embed, attention_ques)
        attention_feat_ques = torch.sum(attention_feat_ques, dim=-1)
        # print(attention_feat_ques.shape)

        return attention_feat_vis, attention_feat_ques


class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, init_image_embedding_size, embedding_size, q_vocab_size, a_vocab_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.q_vocab_size = q_vocab_size
        self.a_vocab_size = a_vocab_size
        self.init_image_embedding_size = init_image_embedding_size

        self.q_network = QuestionProcessor(self.q_vocab_size, self.embedding_size)
        self.v_network = ImageProcessor(self.init_image_embedding_size, self.embedding_size)
        self.w_attention = Attention(self.embedding_size, 512)
        self.p_attention = Attention(self.embedding_size, 512)
        self.q_attention = Attention(self.embedding_size, 512)

        self.attention_word_w = nn.Linear(self.embedding_size, self.embedding_size)
        self.attention_phrase_w = nn.Linear(2*self.embedding_size, self.embedding_size)
        self.attention_question_w = nn.Linear(2*self.embedding_size, self.embedding_size)

        self.classifier = nn.Linear(self.embedding_size, self.a_vocab_size)

        self.activation = nn.Tanh()
        # self.softmax = nn.Softmax(dim=-1)

        # self.word_embedding = nn.Linear(self.q_vocab_size, self.q_embedding_size, bias=False)

        # self.phrase_unigram = nn.Conv1d(self.q_embedding_size, )
        # self.classifier = nn.Linear(self.img_feat_size + self.q_embedding_size, self.a_vocab_size, bias=False)

    def forward(self, image_encoding, question_encoding, question_encoding_oh, question_length):
        (w_embed, p_embed, q_embed) = self.q_network(question_encoding, question_length)
        image_embed = self.v_network(image_encoding)

        vis_feat_word, ques_feat_word = self.w_attention(w_embed, image_embed) # Returns transposed axes for question features
        vis_feat_phrase, ques_feat_phrase = self.p_attention(p_embed, image_embed)
        vis_feat_question, ques_feat_question = self.q_attention(q_embed, image_embed)

        res1 = self.attention_word_w(vis_feat_word + ques_feat_word)
        res1 = self.activation(res1)

        temp = vis_feat_phrase + ques_feat_phrase
        res2 = self.attention_phrase_w(torch.cat([temp, res1], dim=-1))
        res2 = self.activation(res2)

        temp = vis_feat_question + ques_feat_question
        res3 = self.attention_question_w(torch.cat([temp, res2], dim=-1))
        res3 = self.activation(res3)

        logits = self.classifier(res3)
        return logits
