import torch

import json
import glob
from PIL import Image


# 먼저 Vocab을 선언 할때부터 word2idx와 idx2word를 만들어 놓고 시작
class Vocabulary:
    def __init__(self, json_root_dir, num_caption, num_fre, tokenize_func = None):
        self.PAD_token = 0
        self.EOS_token = 1
        self.SOS_token = 2
        self.UNK_token = 3

        if tokenize_func is None:
            self.tokenize_func = lambda x: x.split(" ")
        else:
            self.tokenize_func = tokenize_func

        with open(json_root_dir, "r") as f:
            origin_json_file = sorted(json.load(f)["images"], key=lambda x: x["file_name"])

        self.json_file = []
        for idx, file in enumerate(origin_json_file):
            if len(file["captions"]) == 5:  # caption갯수가 5개인것만 쓴다.
                self.json_file.append(file)

        check_frequency = {}

        self.max_sentence_len = 0

        for file in self.json_file:
            for cap in file["captions"][:num_caption]:  # train으로 caption 몇개를 쓸건지에 따라 vocab도 달라진다.
                tokens = self.tokenize_func(cap.lower())
                if self.max_sentence_len < len(tokens):
                    self.max_sentence_len = len(tokens)
                for token in tokens:
                    if token not in check_frequency:
                        check_frequency[token] = 0
                    else:
                        check_frequency[token] += 1

        self.word2idx = {"[PAD]": self.PAD_token, "[EOS]": self.EOS_token, "[SOS]": self.SOS_token, "[UNK]" : self.UNK_token}
        self.idx2word = {0: "[PAD]", 1: "[EOS]", 2: "[SOS]", 3: "[UNK]"}
        self.num_words = len(self.word2idx)


        for (key, value) in check_frequency.items():
            if value >= num_fre:
                self.word2idx[key] = self.num_words
                self.idx2word[self.num_words] = key
                self.num_words += 1



class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_root_dir, json_root_dir, transform, num_caption, num_fre, tokenize_func=None, train = True, val_ratio = 0.2):
        super(Dataset, self).__init__()
        if tokenize_func == None:
            self.tokenize_func = lambda x: x.split(" ")
        else:
            self.tokenize_func = tokenize_func
        self.origin_img_paths = sorted(glob.glob(img_root_dir + "/*.jpg"))
        with open(json_root_dir, "r") as f:
            self.origin_json_file = sorted(json.load(f)["images"], key=lambda x: x["file_name"])
        # if the length of the captions is not 5, pop it up.
        self.img_paths = []
        self.json_file = []
        for idx, (label, img) in enumerate(zip(self.origin_json_file, self.origin_img_paths)):
            if len(label["captions"]) == 5:
                self.img_paths.append(self.origin_img_paths[idx])
                self.json_file.append(self.origin_json_file[idx])
        self.vocab = Vocabulary(json_root_dir, num_caption, num_fre)

        # make captions
        self.captions = []
        for i in range(len(self.json_file)):
            captions = []
            for caption in self.json_file[i]["captions"][:num_caption]:
                captions.append(caption.lower())
            self.captions.append(captions)

        n_val = int(len(self.img_paths) * val_ratio)
        if train:
            self.img_paths = self.img_paths[n_val:]
            self.captions = self.captions[n_val:]
        else:
            self.img_paths = self.img_paths[:n_val]
            self.captions = self.captions[:n_val]
        self.train = train
        self.transform = transform
        #print(self.transform)
        #print("img path : {} , captions : {}".format(len(self.img_paths), len(self.captions)))
        #print("longest sentence length : {}".format(self.vocab.max_sentence_len))
        #print("vocab size : {}".format(self.vocab.num_words))
        #print()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # make image
        img = Image.open(self.img_paths[idx])
        img = self.transform(img)
        # make captions including token padding with [PAD]
        captions = []
        caption_length = []
        for caption in self.captions[idx]:
            temp_caption = [self.vocab.word2idx[token] if token in self.vocab.word2idx else self.vocab.UNK_token for token in self.tokenize_func(caption)]
            temp_caption.append(self.vocab.EOS_token)
            temp_caption.insert(0, self.vocab.SOS_token)
            caption_length.append(len(temp_caption))
            if len(temp_caption) - 2 < self.vocab.max_sentence_len:
                temp_caption.extend([self.vocab.PAD_token] * (self.vocab.max_sentence_len - len(temp_caption) + 2))
            captions.append(temp_caption)
        all_captions = torch.LongTensor(captions)
        captions = torch.Tensor(captions).long()
        caption_length = torch.LongTensor([caption_length])

        if self.train:
            return img, captions, caption_length
        else:

            return img, captions, caption_length, all_captions
        # return img, self.captions[idx]
