# code based on https://dcase.community/challenge2022/task-language-based-audio-retrieval with some modifications
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pickle

import h5py
import numpy as np
import pandas as pd
import yaml
"""
Helper class to instantiate a vocabulary, based on DCASE:
https://github.com/xieh97/dcase2022-audio-retrieval
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
        self.audio_feature = audio_feature #? Log mel h5py file
        self.data_df = data_df #? _captions.json file
        self.query_col = query_col #? Column in _captions.json pd that relates to the NER tags
        self.vocabulary = vocabulary #? word embeddings from the vocabulary object

    '''
    Given a a row in captions, it returns (audio_features_for_fid, token_ids_for_fid, info_for_fid)
    '''
    def __getitem__(self, index):
        item = self.data_df.iloc[index]

        audio_feat = torch.as_tensor(self.audio_feature[str(item["fid"])][()])
        query = torch.as_tensor([self.vocabulary(token) for token in item[self.query_col]])

        info = {"cid": item["cid"], "fid": item["fid"], "fname": item["fname"], "caption": item["original"]}

        return audio_feat, query, info

    def __len__(self):
        return len(self.data_df)

#? the DataLoader passes this function a list of tuples, and this function returns a batch 
def collate_fn(data_batch):
    """
    :param data_batch: a list of tensor tuples (audio_feature, query, info).
    :return: (batch_audio_features, batch_query, batch_info)
    """
    audio_feat_batch = []
    query_batch = []
    info_batch = []

    for a, q, i in data_batch:
        audio_feat_batch.append(a)
        query_batch.append(q)
        info_batch.append(i)

    audio_feat_batch, audio_feat_lens = pad_tesnsors(audio_feat_batch)
    query_batch, query_lens = pad_tesnsors(query_batch)

    return audio_feat_batch.float(), audio_feat_lens, query_batch.long(), query_lens, info_batch

#? realistically padding is only needed for the tokens, as there can be larger sentences than others
def pad_tesnsors(tensor_list):
    tensor_lens = [tensor.shape for tensor in tensor_list]

    dim_max_lens = tuple(np.max(tensor_lens, axis=0))

    tensor_lens = np.array(tensor_lens)[:, 0]

    ''' 
    len(tensor_list) =  number of tensors, so could be 500
    dim_max_lens = max of (230,), (240,), (180,) = (240,)
    (len(tensor_list),) + dim_max_lens = (500, 240) 
    
    '''
    padded_tensor = torch.zeros((len(tensor_list),) + dim_max_lens)
    for i, t in enumerate(tensor_list):
        end = tensor_lens[i]
        padded_tensor[i, :end] = t[:end]

    return padded_tensor, tensor_lens    


def load_data(config):
    #Loading the audio features
    feats_path = os.path.join(config["hdf5_dir"], config["hdf5_file"])
    audio_feats = h5py.File(feats_path, "r") #? read the h5py file
    print("Load", feats_path)

     # Load pretrained word embeddings
    emb_path = os.path.join(config["pickle_dir"], config["emb_file"])
    with open(emb_path, "rb") as emb_reader:
        word_vectors = pickle.load(emb_reader)
    print("Load", emb_path)

    #Create the vocabualary
    vocabulary = Vocabulary()
    for word in word_vectors:
        if len(vocabulary) == 0:
            #? with this trick the <pad> token is set to index 0, that is why we set the paddings as 0s in the padding function
            vocabulary.add_word("<pad>", np.zeros_like(word_vectors[word])) #? Create pad vector
        vocabulary.add_word(word, word_vectors[word]) #? Add word and the embedding


    #? Looad the data splits
    conf_splits = [config["splits"][split] for split in config["splits"]]
    text_datasets = {}
    for split in conf_splits:
        json_path = os.path.join(config['pickle_dir'], f"{split}_captions.json")
        '''
        columns=["cid", "fid", "fname", "original", "caption", "tokens"],
        cid = caption id
        fid = file id
        fname = file name
        original = original caption
        caption = tokenised caption
        tokens = tokens that are entities and have embeddings are shown, others are <UNK>
        '''
        df = pd.read_json(json_path)
        print('Load', json_path)

        dataset = QueryAudioDataset(audio_feats, df, 'tokens', vocabulary)
        text_datasets[split] = dataset
    

    return text_datasets, vocabulary

'''
if __name__ == "__main__":
    with open("conf.yaml", "rb") as stream:
            conf = yaml.full_load(stream)

    conf_data = conf['data']
    text_datasets, vocabulary = load_data(conf_data)
    #print(list(text_datasets['development'])[0])
    #print(vocabulary.get_weights().shape)
    '''