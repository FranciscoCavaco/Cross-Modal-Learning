# code based on https://dcase.community/challenge2022/task-language-based-audio-retrieval with some modifications
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pickle

import h5py
import numpy as np
import pandas as pd
"""
Helper class to instantiate a vocabulary
"""


class Vocabulary(object):
    def __init__(self):
        self.word2vec = {}
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        self.weights = None

    def add_word(self, word, word_vector):
        if word not in self.word2idx:
            self.word2vec[word] = word_vector
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def get_weights(self):
        for idx in range(self.idx):
            if self.weights is None:
                self.weights = self.word2vec[self.idx2word[idx]]
            else:
                self.weights = np.vstack(
                    (self.weights, self.word2vec[self.idx2word[idx]])
                )

        return self.weights

    def __call__(self, word):
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class QueryAudioDataset(Dataset):

    def __init__(self, audio_feature, data_df, query_col, vocabulary):
        self.audio_feature = audio_feature
        self.data_df = data_df
        self.query_col = query_col
        self.vocabulary = vocabulary

    def __getitem__(self, index):
        item = self.data_df.iloc[index]

        audio_feat = torch.as_tensor(self.audio_feature[str(item["fid"])][()])
        query = torch.as_tensor([self.vocabulary(token) for token in item[self.query_col]])

        info = {"cid": item["cid"], "fid": item["fid"], "fname": item["fname"], "caption": item["original"]}

        return audio_feat, query, info

    def __len__(self):
        return len(self.data_df)


def load_data(config):
    #Loading the audio features
    feats_path = os.path.join(config["hdf5_dir"], config["hdf5_file"])
    audio_feats = h5py.File(feats_path, "r") #? read the h5py file
    print("Load", feats_path)
