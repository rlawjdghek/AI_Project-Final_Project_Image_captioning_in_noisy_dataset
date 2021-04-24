import torch
import torch.nn as nn
from models.attention import Attention_Module

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, attention_size, vocab_size, encoder_size, device, seq_len=54):
        """
        :param embed_size: 임베딩 크기
        :param hidden_size: LSTM hidden size
        :param attention_size: 어텐션 크기
        :param vocab_size: 총 단어 갯수
        :param encoder_size: 인코더에서 나온 마지막 값과 같은 값 2048
        """

        # print("decoder init) embed_size : {}, hidden_size : {}, attention_size : {}, vocab_size : {}, encoder_size : {} "
        #      .format(embed_size, hidden_size, attention_size, vocab_size, encoder_size))

        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.LSTMCell = nn.LSTMCell(embed_size + encoder_size, hidden_size)  # 여기에 들어가는 인풋은 인코더에서 나온것과 어텐션 결과가 concat된것
        self.ec = nn.Linear(encoder_size, hidden_size)  # 인코딩된 이미지를 [BATCH_SIZE * HIDDEN_SIZE]로 만든다.
        self.eh = nn.Linear(encoder_size, hidden_size)  # HIDDEN layer를 [BATCH_SIZE * HIDDEN_SIZE]로 만든다.
        self.sag = nn.Linear(hidden_size, encoder_size)
        self.attention_module = Attention_Module(hidden_size, attention_size, encoder_size)
        self.last_fc = nn.Linear(hidden_size, vocab_size)
        self.device = device

    def forward(self, encoded_img, captions, caption_lengths):
        """
        :param encoded_img: [BATCH_SIZE, encoded_size, encoded_size, encoder_size=2048]
        :param encoded_caption: [BATCH_SIZE, ]
        :param caption_length: [BATCH_SIZE, ]
        """

        encoded_img = encoded_img.reshape(encoded_img.shape[0], -1, encoded_img.shape[-1]) # [BATCH_SIZE * SIZE^2 * ENCODER_SIZE]
        hidden, cell = self.init_hidden_cell_state(encoded_img)

        caption_lengths, caption_length_idx = caption_lengths.squeeze(1).sort(dim=0, descending = True)
        encoded_img = encoded_img[caption_length_idx]  # 패딩이 길어서 계산 손해 낮추기
        captions = captions[caption_length_idx]
        embedded_captions = self.embedding(captions)  # [BATCH_SIZE * SEQ_LEN] = >[BATCH_SIZE * SEQ_LEN * EMBED_SIZE]
        pred_length = (caption_lengths-1).tolist()

        # pad = 0이므로 pred에서 값이 안바뀐 곳은 자동적으로 pad라 할 수 있다.
        preds = torch.zeros((embedded_captions.shape[0], max(pred_length), self.vocab_size)).to(self.device)  # [BATCH_SIZE * (VALID_SEQ_LEN - 1) * VOCAB_SIZE]
        coefs = torch.zeros((embedded_captions.shape[0], max(pred_length), encoded_img.shape[1])).to(self.device)    # [BATCH_SIZE * (VALID_SEQ_LEN - 1) * (SIZE^2)]
        # 이제 pred_length만큼 반복해서 하나씩 예측. pad는 예측 안하기 위해서
        for seq_idx in range(max(pred_length)):
            num_not_pad = 0 # 배치에서 몇개가 현재 예측하는 위치에서 padding이 아닌지.
            for length in pred_length:
                if length > seq_idx:
                    num_not_pad += 1
            attentioned_encoder_output, coef = self.attention_module(encoded_img[:num_not_pad], hidden[:num_not_pad])
            gs = torch.sigmoid(self.sag(hidden[:num_not_pad]))
            attentioned_encoder_output = attentioned_encoder_output * gs  # show and tell paper 참조.
            new_inputs = torch.cat([embedded_captions[:num_not_pad, seq_idx, :] , attentioned_encoder_output], dim=1) # [BATCH_SIZE * (EMBED_SIZE + 2048)]
            hidden, cell = self.LSTMCell(new_inputs, (hidden[:num_not_pad], cell[:num_not_pad])) # cell 이므로 hidden이 곧 output
            preds[:num_not_pad, seq_idx, : ] = self.last_fc(hidden)  # [BATCH_SIZE * (SIZE^2)]

        return pred_length, captions, preds, coefs

    def init_hidden_cell_state(self, encoded_img):
        encoded_img = encoded_img.mean(dim=1)  # [BATCH_SIZE * IMG_SIZE^2 * 2048] => [BATCH_SIZE, 2048]
        # print("init hidden cell state) encoded_img : {}".format(encoded_img.shape))
        hidden = self.eh(encoded_img) # [BATCH_SIZE, ]
        cell = self.ec(encoded_img)
        return hidden, cell