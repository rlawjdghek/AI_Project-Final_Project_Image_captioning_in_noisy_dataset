import torch.nn as nn
import torch.nn.functional as F

class Attention_Module(nn.Module):
    def __init__(self, hidden_size, attention_size, encoder_size):
        super(Attention_Module, self).__init__()
        self.encoder_attention = nn.Linear(encoder_size, attention_size)
        self.decoder_attention = nn.Linear(hidden_size, attention_size)
        self.attention = nn.Linear(attention_size, 1)

    def forward(self, encoder_output, hidden):
        """
        :param encoder_output: [BATCH_SIZE, (SIZE^2), 2048]
        :param hidden:
        :return:
        """
        # print("attention forward) encoder output : {}, hidden : {}".format(encoder_output.shape, hidden.shape))
        encoder_att_output = self.encoder_attention(encoder_output) # [BATCH_SIZE x (SIZE^2) x ATTENTION_SIZE]
        decoder_att_output = self.decoder_attention(hidden).unsqueeze(1)  # [BATCH_SIZE x 1 x ATTENTION_SIZE]
        att_output = F.relu(encoder_att_output + decoder_att_output)  #  [BATCH_SIZE x (SIZE^2) x ATTENTION_SIZE]
        att_output = self.attention(att_output).squeeze(2) # [BATCH_SIZE x (SIZE^2) x 1] => [BATCH_SIZE x (SIZE^2)]
        att_output = F.softmax(att_output, dim=1).unsqueeze(2)  # [BATCH_SIZE x (SIZE^2) x 1]
        attentioned_encoder_output = (att_output * encoder_output).sum(dim=1)  # [BATCH_SIZE x 2048(encoder_size)]
        return attentioned_encoder_output, att_output
