from models.encoder import Encoder
from models.decoder import Decoder
from utils.utils import Dataset
from utils.Logger import Logger

import argparse
import os

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--BATCH_SIZE", type=int, default=64, help="batch size for trianing")
    parser.add_argument("--NUM_EPOCHS", type=int, default=100, help="number of epochs")
    parser.add_argument("--NUM_CAPTIONS", type=int, default=1, choices=range(1, 6),
                        help="number of captions used in training. value must be in 1 to 5")
    parser.add_argument("--INPUT_SIZE", type=int, default=256, help="image size as inputs")
    parser.add_argument("--NUM_LAYERS", type=int, default=1, help="number of layers for LSTM")
    parser.add_argument("--HIDDEN_SIZE", type=int, default=128, help="hidden size for LSTM")
    parser.add_argument("--EMBED_SIZE", type=int, default=128, help="embedding size for tokens")
    parser.add_argument("--ATTENTION_SIZE", type=int, default=128, help="attention size for attention layer")
    parser.add_argument("--ATTENTION_COEF", type=int, default=1, help="attention coefficient")
    parser.add_argument("--LR", type=int, default=0.001, help="learning rate for training")
    parser.add_argument("--NUM_LAYERS_LIST", nargs="+", default=[3, 5, 10, 3], help="number of layers in each resnet block. The length "
                                                                                 "of the list must be 4")
    parser.add_argument("--BIDIRECTION", type=bool, default=False, help="use bidirection or not")
    parser.add_argument("--ENCODER_OUTPUT_SIZE", type=int, default=14, help="encoder output size")
    parser.add_argument("--NUM_FRE", type=int, default=1,
                        help="The number of times that must appear in order to be used in dictionary")
    parser.add_argument("--MODEL_SAVE_PATH", type=str, default="./saved_model", help="directory for saving models")
    parser.add_argument("--MODEL_NAME", type=str, default="_model.pth", help="name for saved model")
    parser.add_argument("--gpus", default="0", help="gpu number")
    parser.add_argument("--log_path", default="./train.txt", help="log path")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    logger = Logger()
    logger.open("./log/" + args.log_path)

    if not os.path.isdir(args.MODEL_SAVE_PATH):
        os.makedirs(args.MODEL_SAVE_PATH)

    transform = transforms.Compose([
        transforms.Resize((args.INPUT_SIZE, args.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = Dataset("./data", "utils/captions.json", transform, args.NUM_CAPTIONS, num_fre=args.NUM_FRE, train=True)
    validation_dataset = Dataset("./data", "utils/captions.json", transform, args.NUM_CAPTIONS, num_fre=args.NUM_FRE, train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.BATCH_SIZE, drop_last=True, num_workers = 4)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, drop_last=True, num_workers = 4)

    VOCAB_SIZE = train_dataset.vocab.num_words
    SEQ_LEN = train_dataset.vocab.max_sentence_len

    encoder = Encoder(args.ENCODER_OUTPUT_SIZE).to(device)
    decoder = Decoder(embed_size = args.EMBED_SIZE, hidden_size=args.HIDDEN_SIZE, attention_size=args.ATTENTION_SIZE, vocab_size=VOCAB_SIZE,
                      encoder_size=2048, device=device, seq_len = SEQ_LEN+2).to(device)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr = args.LR)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr = args.LR)

    criterion = nn.CrossEntropyLoss()  # 나중에 loss 계산 할때 패딩은 모두 없앨것이므로 ignore index를 설정하지 않는다

    train_losses = []
    validation_losses = []

    for epoch in range(args.NUM_EPOCHS):
        train_loss = 0
        validation_loss = 0
        encoder.train()
        decoder.train()
        for idx, (img, caption_5, caption_lengths_5) in enumerate(train_loader):
            origin_img = img
            for i in range(args.NUM_CAPTIONS):
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                img = origin_img.to(device)
                caption = caption_5[:, i, :].to(device)
                caption_lengths = caption_lengths_5[:, :, i].to(device)


                img = encoder(img)
                pred_length, captions, preds, coefs = decoder(img, caption, caption_lengths) #  preds => [BATCH_SIZE * (현재 배치에서 유효한 가장 긴 길이) * VOCAB_SIZE]


                target = captions[:, 1:] # 이전의 start를 뺀 것과 같다. 단 길이별로 정렬되어있음 [BATCH_SIZE * (SEQ_LEN + 1)]

                preds = nn.utils.rnn.pack_padded_sequence(preds, pred_length, batch_first=True)  # 패딩을 모두 없애고 한줄로 핀다. 즉 target과 비교될때 CNN처럼 2차원 비교를 한다고
                # 생각, 계산 최소화. [(유효한 길이 합) * VOCAB_SIZE] 주의할 점은 테스트뽑을때 배치별로 붙으므로 한줄을 읽으려면 stride를 BATCH_SIZE로 줘야한다.
                target = nn.utils.rnn.pack_padded_sequence(target, pred_length, batch_first=True)  # 마찬가지로 패딩을 모두 없애고 한줄로 핀다. [유효한 길이 합]


                loss = criterion(preds.data, target.data)
                loss = loss + args.ATTENTION_COEF * ((1.0 - coefs.sum(dim=1)) ** 2).mean()  # regularization
                train_loss = train_loss + loss
                loss.backward()
                decoder_optimizer.step()
                encoder_optimizer.step()  # GAN과 마찬가지로 역순

            train_losses.append(train_loss)
            if idx % 100 ==0:
                logger.write("batch  : [{}/{}]\n".format(idx , len(train_loader)))
        logger.write("train_loss : {}\n".format(train_loss / (len(train_loader) * args.NUM_CAPTIONS)))

        logger.write("-"*10 + "validation" + "-"*10 + "\n")
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for idx, (img, caption_5, caption_lengths_5, _) in enumerate(validation_loader):
                origin_img = img

                for i in range(args.NUM_CAPTIONS):
                    img = origin_img.to(device)
                    caption = caption_5[:, i, :].to(device)
                    caption_lengths = caption_lengths_5[:,:,i].to(device)


                    img = encoder(img)
                    pred_length, captions, preds, coefs = decoder(img, caption, caption_lengths) #  preds => [BATCH_SIZE * (현재 배치에서 유효한 가장 긴 길이) * VOCAB_SIZE]


                    target = captions[:, 1:] # 이전의 start를 뺀 것과 같다. 단 길이별로 정렬되어있음 [BATCH_SIZE, SEQ_LEN(53)]

                    preds = nn.utils.rnn.pack_padded_sequence(preds, pred_length, batch_first=True)
                    target = nn.utils.rnn.pack_padded_sequence(target, pred_length, batch_first=True)

                    loss = criterion(preds.data, target.data)
                    loss = loss + args.ATTENTION_COEF * ((1.0 - coefs.sum(dim=1)) ** 2).mean()  #  regularization
                    validation_loss += loss

            validation_losses.append(validation_loss)
            logger.write("val_loss : {}\n".format(validation_loss / (len(validation_loader) * args.NUM_CAPTIONS)))
        if epoch % 5 == 0:
            torch.save(encoder.state_dict(), os.path.join(args.MODEL_SAVE_PATH, "{}".format(epoch) + "encoder" + args.MODEL_NAME))
            torch.save(decoder.state_dict(), os.path.join(args.MODEL_SAVE_PATH, "{}".format(epoch) + "decoder" + args.MODEL_NAME))


