from models.encoder import Encoder
from models.decoder import Decoder
from utils.utils import Vocabulary
from utils.Logger import Logger


import argparse
import os
from glob import glob
from PIL import Image
import json

import torch
from torchvision import transforms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--BATCH_SIZE", type=int, default=64, help="batch size for trianing")
    parser.add_argument("--NUM_CAPTIONS", type=int, default=1, choices=range(1, 6),
                        help="number of captions used in training. value must be in 1 to 5")
    parser.add_argument("--INPUT_SIZE", type=int, default=256, help="image size as inputs")
    parser.add_argument("--HIDDEN_SIZE", type=int, default=128, help="hidden size for LSTM")
    parser.add_argument("--ENCODER_OUTPUT_SIZE", type=int, default=14, help="encoder output size")
    parser.add_argument("--EMBED_SIZE", type=int, default=128, help="embedding size for tokens")
    parser.add_argument("--ATTENTION_SIZE", type=int, default=128, help="attention size for attention layer")
    parser.add_argument("--BIDIRECTION", type=bool, default=False, help="use bidirection or not")
    parser.add_argument("--NUM_FRE", type=int, default=1,
                        help="The number of times that must appear in order to be used in dictionary")
    parser.add_argument("--NUM_TOP_PROB", type=int, default=5, help="how much sentences are predicted")
    parser.add_argument("--MAX_SENTENCE_LEN", type=int, default=100, help="maximum sentence length")
    parser.add_argument("--ENCODER_MODEL_LOAD_PATH", type=str, required=True, help="encoder model path")
    parser.add_argument("--DECODER_MODEL_LOAD_PATH", type=str, required=True, help="decoder model path")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--log_path", default="./test.txt", help="log path")
    parser.add_argument("--test_img_dir", type=str, default="./test_img")
    args = parser.parse_args()

    IMG_PATH = sorted(glob(os.path.join(args.test_img_dir, "*.jpg")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    logger = Logger()
    logger.open("./log/" + args.log_path)

    transform = transforms.Compose([
        transforms.Resize((args.INPUT_SIZE, args.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    vocab = Vocabulary("./captions.json", args.NUM_CAPTIONS, num_fre=args.NUM_FRE)
    VOCAB_SIZE = vocab.num_words
    SEQ_LEN = vocab.max_sentence_len

    encoder = Encoder(args.ENCODER_OUTPUT_SIZE)
    decoder = Decoder(embed_size=args.EMBED_SIZE, hidden_size=args.HIDDEN_SIZE, attention_size=args.ATTENTION_SIZE,
                      vocab_size=VOCAB_SIZE, encoder_size=2048,
                      device=device, seq_len=SEQ_LEN + 2)
    encoder.load_state_dict(torch.load(args.ENCODER_MODEL_LOAD_PATH))
    decoder.load_state_dict(torch.load(args.DECODER_MODEL_LOAD_PATH))


    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    result_json = {"images" : []}
    for path in IMG_PATH:
        img_name = path.split("/")[-1]
        img = Image.open(path)
        img = transform(img).unsqueeze(0).to(device)  # [BATCH_SIZE(1) * CHANNEL * INPUT_SIZE * INPUT_SIZE]

        num_sentence = args.NUM_TOP_PROB
        top_prev_prob = torch.zeros((num_sentence, 1)).to(device)
        words = torch.Tensor([vocab.SOS_token]).long().expand(num_sentence, -1).to(
            device)  # [num_sentence * 1]  이 단어들이 계속 가면서 다음 단어로 바뀜 처음에는 모두 [SOS]
        sentences = words  # 문장들이 저장되는 곳, 나중에 여기서 최고 확률의 [EOS]를 뽑는다.
        end_sentences = []  # NUM_TOP_PROB갯수만큼의 끝난 문장들만 모아둠
        end_sentences_score = []  # NUM_TOP_PROB갯수만큼의 끝난 문장들의 [END]가 떴을때 score를 모아둠.
        len_sentence = 1
        with torch.no_grad():
            encoded_img = encoder(img)
            encoded_img = encoded_img.reshape(1, -1, 2048).expand(num_sentence, -1,
                                                                  -1)  # [num_sentence * ENCODER_OUTPUT_SIZE^2 * 2048]
            hidden, cell = decoder.init_hidden_cell_state(encoded_img)
            is_first = True
            # decoder 시작. SOS를 제외하고 너무 길지 않게만 반복, 여러개 문장으로 시작해서 최종 END의
            while len_sentence < args.MAX_SENTENCE_LEN:
                embedded_words = decoder.embedding(words).squeeze(1)  # [num_sentence * EMBED_SIZE]
                attentioned_encoder_output, _ = decoder.attention_module(encoded_img, hidden)  # 이제부터는 그냥 무조건 다 PAD가 아니다.
                gs = torch.sigmoid(decoder.sag(hidden))
                attentioned_encoder_output = attentioned_encoder_output * gs
                new_inputs = torch.cat([embedded_words, attentioned_encoder_output], dim=1)
                hidden, cell = decoder.LSTMCell(new_inputs, (hidden, cell))
                preds = decoder.last_fc(hidden)  # [(<num_sentence) * VOCAB_SIZE] => train과 마찬가지로 end가 나오면 배치가 작아짐.

                preds = preds + top_prev_prob.expand_as(preds)  # 이전 확률과 새롭게 예측한 확률을 더해서 계속 누적해나감.

                if is_first:  # 만약 처음이면 5개의 모든 결과가 같으므로 처음 것에서만 5개를 뽑음
                    top_prev_prob, top_prob_idx = preds[0].topk(num_sentence, dim=0, largest=True, sorted=True)
                    idx_where_sentence = torch.LongTensor([0, 0, 0, 0, 0])  # 맨처음엔 모두 [SOS]이므로 0번째 문장에서 뽑았다고 해도된다.
                    is_first = False
                else:  # 아니면 이제부턴 5개의 문장에서 예측한 최고 확률 5개를 뽑음.
                    top_prev_prob, top_prob_idx = preds.reshape(-1).topk(num_sentence, dim=0, largest=True,
                                                                         sorted=True)  # num_sentence만큼 뽑는다. 그리고 뽑은 문장 위치를 오름차순으로 저장
                    idx_where_sentence = top_prob_idx // preds.shape[1]  # 그 다음부터는 몇번 문장에서부터 뽑혔는지 구한다. 즉 이문장들이 살아남는다.
                real_prob_idx = (top_prob_idx % preds.shape[1]).unsqueeze(
                    1)  # 첫번째 문장이 아닌 곳에서 뽑힌 인덱스는 넘위를 넘어가므로 재조정 [num_sentence * 1]
                sentences = sentences[idx_where_sentence]  # 최고 확률을 뽑은 문장에 저장
                sentences = torch.cat([sentences, real_prob_idx], dim=1)  # [num_sentence * (len_sentence+1)]

                idx_end_sentence = []
                idx_not_end_sentence = []
                for idx, word in enumerate(real_prob_idx.squeeze(1).tolist()):
                    if vocab.EOS_token == word:
                        idx_end_sentence.append(idx)
                    else:
                        idx_not_end_sentence.append(idx)

                if len(idx_end_sentence) != 0:  # 만약 끝난 문장이 있다면
                    end_sentences.extend(sentences[idx_end_sentence].tolist())
                    end_sentences_score.extend(top_prev_prob[idx_end_sentence].tolist())  # [END]가 떳을때의 score
                num_sentence = num_sentence - len(idx_end_sentence)  # 끝난 문장갯수만큼 총 문장갯수를 빼줌.
                len_sentence += 1

                # 끝나지 않은 문장들로만 구성 업데이트
                sentences = sentences[idx_not_end_sentence]
                hidden = hidden[idx_where_sentence[idx_not_end_sentence]]
                cell = cell[idx_where_sentence[idx_not_end_sentence]]
                encoded_img = encoded_img[idx_where_sentence[idx_not_end_sentence]]
                top_prev_prob = top_prev_prob[idx_not_end_sentence].unsqueeze(1)
                words = real_prob_idx[idx_not_end_sentence]
                #print(sentences)
                if num_sentence == 0:
                    break
        if len(end_sentences_score)==0:
            # 아무것도 안뽑힐때 그냥 0번째 보는것.
            for idx, score in enumerate(end_sentences_score):
                if max_end_score < score:
                    max_end_score = score
                    max_end_score_idx = idx
            result = sentences.tolist()[max_end_score_idx]
            result_sentence = [vocab.idx2word[idx] for idx in result]
            sentence = " ".join(result_sentence[1:-1])
            temp = {"file_name" : img_name, "captions" : sentence}
            result_json["images"].append(temp)
        else:
            # 뽑혔을때 최고 score로 판단
            max_end_score = 0
            max_end_score_idx = 0
            for idx, score in enumerate(end_sentences_score):
                if max_end_score < score:
                    max_end_score = score
                    max_end_score_idx = idx
            result = end_sentences[max_end_score_idx]
            result_sentence = [vocab.idx2word[idx] for idx in result]
            sentence = " ".join(result_sentence[1:-1])
            temp = {"file_name" : img_name, "captions" : sentence}
            result_json["images"].append(temp)

    with open("./result.json", "w") as f:
        json.dump(result_json, f)


