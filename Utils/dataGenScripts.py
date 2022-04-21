import os
from re import S
from typing import List, Union


from numpy import isin, number, spacing
import numpy as np
import librosa
from torch import AnyType
import yaml
import h5py
import glob
from mutagen.wave import WAVE
from tqdm import tqdm
import pickle
import pandas as pd

#! NLP
import gensim.downloader as downloader
import nltk
from nltk.corpus import wordnet
import re
import contractions
import string

nltk.download("maxent_ne_chunker")
nltk.download("words")


class Typing_Utils:
    # ? Check if list val is of type types
    def list_checker(self, val: List, _types: List[type]):

        not_error = True
        if len(val) == 0:
            not_error = False

        for element in val:
            if not isinstance(element, tuple([*_types])):
                not_error = False
                break
        return not_error


"""
Generates:
- Caption info
- audio_info
- vocab_info

"""


class Preprocess:
    def __init__(self) -> None:
        self.audio_fids, self.audio_fnames, self.audio_durations = {}, {}, {}
        # ? Pretrained word2vec model
        word2vec = "word2vec-google-news-300"
        self.embedding_model = downloader.load(word2vec)
        # ? Code inspired by https://dcase.community/challenge2022/task-language-based-audio-retrieval
        self.stopwords = nltk.corpus.stopwords.words("english")  # Get stopwords
        self.UNK_token = "<UNK>"  # ? Unknown

    """
    This downloads the embedding model and applies IOB chunking along with the POS tag
    """

    def __word_processor(self, _text):

        # ? nltk pos is trained using the Treebank dataset
        # ? So we can translate from Treebank to wordnet
        # https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
        wordnet_pos_tag = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
        }

        # ? Local embedder
        def __embedder(word):
            try:
                return self.embedding_model.get_vector(word)
            except KeyError:
                return None

        self.__embedder = __embedder  # ? Store for later use
        #! Cleaning the data
        # TODO: Remove stopwords? Lemmatise? Implement slang option in the contractions
        text = re.sub(r"\s+", " ", _text)  # ? Removing repeated spaces
        text = contractions.fix(text)  # ? Expand contractions
        text = re.sub(r"-+", " ", text)  # ? Remove dashes, replace with spaces
        text = "".join(
            [i for i in text if i not in string.punctuation]
        )  # ? remoe punctuation

        # ? tokenizing & POS
        tokens = nltk.word_tokenize(text)
        tokens_tag = nltk.pos_tag(tokens)

        # ? chunking
        # https://www.guru99.com/pos-tagging-chunking-nltk.html
        # https://www.nltk.org/book/ch07.html
        chunked_tree = nltk.ne_chunk(tokens_tag, binary=True)
        tokens_tag = nltk.tree2conlltags(
            chunked_tree
        )  # Convert to word, POStag, IOBtag https://stackoverflow.com/questions/40879520/nltk-convert-a-chunked-tree-into-a-list-iob-tagging

        words = list()

        # ? filter out the relevant words
        for token, pos_tag, iob_tag in tokens_tag:
            if (
                iob_tag != "O"
            ):  # ? This means that it is not outside thus it must be part of a NE
                words.append(self.UNK_token)

            elif pos_tag in [
                "POS"
            ]:  # ? If it is a possessive ending (POS = part of speech tag) https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html#:~:text=17.,Possessive%20ending
                pass

            elif token in string.punctuation:
                pass

            elif __embedder(token) is not None:  # ? if the token has an embedding
                words.append(token)

            else:
                # ? we now try to lemmatise and stem the word to find an embedding
                lemmatizer = nltk.WordNetLemmatizer()
                # ? Since our POS are in Treebank form we try to see if we can match it with a wordnet POS, if not we assume its a noun
                lemma = lemmatizer.lemmatize(
                    token, pos=wordnet_pos_tag.get(pos_tag[0], wordnet.NOUN)
                )

                if (
                    __embedder(lemma) is not None
                ):  # ? if we find an embeddin for the lemma store it
                    words.append(lemma)
                else:
                    # ? Try to stem it further to possibly discover an embedding
                    stemmer = nltk.SnowballStemmer(language="english")
                    root = stemmer.stem(lemma)

                    if __embedder(root) is not None:
                        words.append(root)
                    else:
                        words.append(self.UNK_token)  # ? No embedding found
        return text, words

    # ? Seperate out the audio files
    def __audio_seperator(self):
        with open("conf.yaml", "rb") as stream:
            conf = yaml.full_load(stream)

        conf_data = conf["data"]
        conf_splits = [conf_data["splits"][split] for split in conf_data["splits"]]

        fid_count = 0
        for split in conf_splits:

            fids, fnames, durations = {}, [], []
            # Glob is to get all the file name
            for fpath in tqdm(
                glob.glob(
                    r"{}/*.wav".format(os.path.join(conf_data["dataset_dir"], split))
                ),
                desc=split,
            ):
                # ? From baseline code https://dcase.community/challenge2022/task-language-based-audio-retrieval
                try:
                    clip = WAVE(fpath)
                    if clip.info.length > 0.0:
                        fid_count += 1

                        fname = os.path.basename(fpath).strip()

                        fids[fname] = fid_count
                        fnames.append(fname)
                        durations.append(clip.info.length)

                except:
                    print("Error audio file: {}.".format(fpath))

                self.audio_fids[split] = fids
                self.audio_fnames[split] = fnames
                self.audio_durations[split] = durations

        # This object is then saved using pickle
        with open(
            os.path.join(conf_data["pickle_dir"], "audio_info.pkl"), "wb"
        ) as store:
            pickle.dump(
                {
                    "audio_fids": self.audio_fids,
                    "audio_fnames": self.audio_fnames,
                    "audio_durations": self.audio_durations,
                },
                store,
            )
        print("Saved audio info")

    def __caption_seperator(
        self,
    ):  # ? base code https://dcase.community/challenge2022/task-language-based-audio-retrieval
        if not len(self.audio_fids):
            raise NotImplementedError(
                "Make sure to run __audio_seperator() before this function as self.audio_fids is not set."
            )
        with open("conf.yaml", "rb") as stream:
            conf = yaml.full_load(stream)

        conf_captions = conf["captions"]
        conf_data = conf["data"]
        conf_splits = [conf_data["splits"][split] for split in conf_data["splits"]]

        cid_count = 0
        for split in conf_splits:
            cap_file = f"{conf_captions['caption_prefix']}{split}.csv"  # ? File name for current split
            captions = pd.read_csv(
                os.path.join(conf_captions["captions_dir"], cap_file)
            )
            split_fids = self.audio_fids[split]
            audio_captions = list()
            for i in tqdm(captions.index, desc=f"{split}_captions"):
                # ? Extracting caption information
                fname = captions.iloc[i].file_name.strip()
                c1 = captions.iloc[i].caption_1
                c2 = captions.iloc[i].caption_2
                c3 = captions.iloc[i].caption_3
                c4 = captions.iloc[i].caption_4
                c5 = captions.iloc[i].caption_5

                fid = split_fids[fname]

                for cap in [c1, c2, c3, c4, c5]:
                    cid_count += 1

                    text, words = self.__word_processor(cap)

                    # cid, fid, fname, original, caption, tokens
                    audio_captions.append([cid_count, fid, fname, cap, text, words])

            # ? Save JSON
            audio_captions = pd.DataFrame(
                data=audio_captions,
                columns=["cid", "fid", "fname", "original", "caption", "tokens"],
            )
            audio_captions.to_json(
                os.path.join(
                    conf_captions["json_dir"], "{}_captions.json".format(split)
                )
            )
            print("Saved", "{}_captions.json".format(split))

    def __vocab_info(self):
        # ?From https://github.com/xieh97/dcase2022-audio-retrieval
        if not len(self.audio_fids):
            raise NotImplementedError(
                "Make sure to run __audio_seperator() before this function as self.audio_fids is not set."
            )
        vocabulary = set()
        word_bags = {}
        split_infos = {}

        with open("conf.yaml", "rb") as stream:
            conf = yaml.full_load(stream)

        conf_data = conf["data"]
        conf_splits = [conf_data["splits"][split] for split in conf_data["splits"]]

        for split in tqdm(conf_splits, desc=f"Vocab Info"):
            captions = pd.read_json(
                os.path.join(conf_data["pickle_dir"], "{}_captions.json".format(split))
            )
            split_fids = self.audio_fids[split]

            num_clips = len(split_fids)
            num_captions = captions.caption.size

            bag = []
            for tokens in captions["tokens"]:
                bag.extend(tokens)
                vocabulary = vocabulary.union(tokens)

            num_words = len(bag)
            word_bags[split] = bag
            split_infos[split] = {
                "num_clips": num_clips,
                "num_captions": num_captions,
                "num_words": num_words,
            }

        # Save vocabulary
        with open(
            os.path.join(conf_data["pickle_dir"], "vocab_info.pkl"), "wb"
        ) as store:
            pickle.dump(
                {
                    "vocabulary": vocabulary,
                    "word_bags": word_bags,
                    "split_infos": split_infos,
                },
                store,
            )
        print("Saved vocabulary info")

    #? Converts the vocabulary from vocab_info (which came from the NER from the _captions.json)
    #? into an embedded form
    def __embedding_gen(self):
        with open("conf.yaml", "rb") as stream:
            conf = yaml.full_load(stream)

        if not self.__embedder:
            raise NotImplementedError(
                "Make sure to run __word_processor() before this function as self.__embedder is not set."
            )

        word_embs = {}
        emb_matrix, emb_shape = None, None

        conf_data = conf["data"]
        try:
            with open(
                os.path.join(conf_data["pickle_dir"], "vocab_info.pkl"), "rb"
            ) as store:
                vocab_info = pickle.load(store)
                vocabulary = vocab_info["vocabulary"]
        except:
            print("Problem loading file: vocab_info.pkl, does it exist?")

        # Generate the word embeddings from the preprocessed tokens
        for word in tqdm(vocabulary, desc="Embeddings Gen"):
            if word != self.UNK_token:
                word_embs[word] = self.__embedder(word)

                if emb_shape is None:
                    emb_shape = word_embs[word].shape  # ? set embedding size

                if emb_matrix is None:
                    emb_matrix = word_embs[word]  # ? create embedding matrix
                else:
                    emb_matrix = np.vstack((emb_matrix, word_embs[word]))

        mean, std = np.mean(emb_matrix, axis=0), np.std(
            emb_matrix, axis=0
        )  # mean & std = (300,)

        # Generate UNK_token embedding
        dot_product = 1.0
        UNK_emb = np.zeros_like(mean)

        # ? Generate the least similar embedding
        while np.all(
            np.abs(dot_product) > 0.01
        ):  # ? Check if one is much more different
            UNK_emb = mean + std * np.random.randn(
                emb_shape[0]
            )  # ? Go furthest away from the mean
            dot_product = np.dot(emb_matrix, UNK_emb)  # ? perform row wise dot product
            print(dot_product.shape)

        word_embs[self.UNK_token] = UNK_emb

        # Save pretrained embeddings
        with open(
            os.path.join(conf_data["pickle_dir"], conf_data["emb_file"]), "wb"
        ) as store:
            pickle.dump(word_embs, store)
        print("Saved pretrained embeddings info")

    # ? Seperate audio files
    def __call__(self):
        self.__audio_seperator()
        self.__caption_seperator()
        self.__vocab_info()
        self.__embedding_gen()


"""
This calculates the log mel spectogram and as a function to place it into an h5py format

Generates the audio_logmel file
"""


class Utils:
    class Audio:
        def __init__(self) -> None:
            self.audio_params = {
                "sample_rate": 44100,
                "window_length_secs": 0.025,
                "hop_length_secs": 0.010,
                "num_mels": 128,
                "fmin": 12.0,
                "fmax": 8000,
                "log_offset": 0.0,
            }

        def set_audio_params(
            self,
            sample_rate,
            window_length_secs,
            hop_length_secs,
            num_mels,
            fmin,
            fmax,
            log_offset,
        ):
            self.audio_params["sample_rate"] = sample_rate
            self.audio_params["window_length_secs"] = window_length_secs
            self.audio_params["hop_length_secs"] = hop_length_secs
            self.audio_params["num_mels"] = num_mels
            self.audio_params["fmin"] = fmin
            self.audio_params["fmax"] = fmax
            self.audio_params["log_offset"] = log_offset

        # Create a log_mel_spectogram
        def __log_mel_spectogram(self, y: List[float], typecheck=True):
            type_check = Typing_Utils()

            if not len(self.audio_params):
                raise NotImplementedError(
                    "audio_params not set, use set_audio_params()"
                )

            if typecheck and not type_check.list_checker(y, [int, float, AnyType]):
                raise TypeError("y must be a list of ints or floats")

            window_length = int(
                round(
                    self.audio_params["sample_rate"]
                    * self.audio_params["window_length_secs"]
                )
            )
            hop_length = int(
                round(
                    self.audio_params["sample_rate"]
                    * self.audio_params["hop_length_secs"]
                )
            )
            fft_length = 2 ** int(
                np.ceil(np.log(window_length) / np.log(2.0))
            )  # Number of bins https://dsp.stackexchange.com/questions/46969/how-can-i-decide-proper-fft-lengthsize#:~:text=Reminder%20%3A%20Bins%20The%20FFT%20size,frequency%20resolution%20of%20the%20window.&text=For%20a%2044100%20sampling%20rate,this%20band%20into%20512%20bins.

            self.mel_spectrogram = librosa.feature.melspectrogram(
                y=y,
                sr=self.audio_params["sample_rate"],
                n_fft=fft_length,
                hop_length=hop_length,
                win_length=window_length,
                n_mels=self.audio_params["num_mels"],
                fmin=self.audio_params["fmin"],
                fmax=self.audio_params["fmax"],
            )

            return np.log(self.mel_spectrogram + self.audio_params["log_offset"])

        # Turn the audio into an h5py file
        def __dir_to_h5py_logmel(self, fids):
            # Load config
            with open("conf.yaml", "rb") as stream:
                conf = yaml.full_load(stream)
            conf_data = conf["data"]

            output_file = os.path.join(conf_data["hdf5_dir"], conf_data["hdf5_file"])

            # Load splits
            conf_splits = [conf_data["splits"][split] for split in conf_data["splits"]]

            with h5py.File(output_file, "w") as feature_store:
                for split in conf_splits:
                    subset_dir = os.path.join(conf_data["dataset_dir"], split)
                    for fpath in tqdm(
                        glob.glob("{}/*.wav".format(subset_dir)),
                        desc=f"{split}_log_mel",
                    ):
                        try:
                            fname = os.path.basename(fpath).strip()
                            fid = fids[split][fname]
                            # Download audio file
                            y, sr = librosa.load(fpath, sr=None, mono=True)
                            # numpy.spacing(1) returns the smallest represented machine number > 1 (machine epsilon)
                            self.set_audio_params(
                                sample_rate=sr,
                                window_length_secs=0.040,
                                hop_length_secs=0.020,
                                num_mels=64,
                                fmin=self.audio_params["fmin"],
                                fmax=self.audio_params["fmax"],
                                log_offset=np.spacing(1),
                            )
                            log_mel = self.__log_mel_spectogram(y, typecheck=False)

                            """
                                Vstack concatinates on the vertical axis, row wise. 
                                so an array of (2,8,6) -> (16, 6)
                                """
                            feat = np.vstack(log_mel).transpose()  # [Time, Mel]
                            feature_store[str(fid)] = feat
                        except:
                            print("Error file: {}.".format(fpath))
            print("Saved logmel data")

        def __call__(self):
            with open("conf.yaml", "rb") as stream:
                conf = yaml.full_load(stream)

            conf_data = conf["data"]

            if os.path.isfile(os.path.join(conf_data["pickle_dir"], "audio_info.pkl")):
                print("there is a file")

            with open(
                os.path.join(conf_data["pickle_dir"], "audio_info.pkl"), "rb"
            ) as store:
                fids = pickle.load(store)["audio_fids"]

            self.__dir_to_h5py_logmel(fids)


"""
Generates all the essential files
"""


class FileGen:
    def __call__(self):
        preprocess = Preprocess()
        utils = Utils().Audio()
        # ? To generate all the necessary files
        preprocess()
        utils()


if __name__ == "__main__":
    fileGen = FileGen()
    fileGen()
