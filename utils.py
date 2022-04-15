import os
from re import S
from typing import List, Union


from numpy import isin, number
import numpy as np
import librosa
import yaml
import h5py
import glob
from mutagen.wave import WAVE
from tqdm import tqdm
import pickle
import pandas as pd


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


class Preprocess:
    def __init__(self) -> None:
        self.audio_fids, self.audio_fnames, self.audio_durations = {}, {}, {}

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

                        fname = os.path.basename(fpath)
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

    def __caption_seperator(self):
        if len(self.audio_fids):
            raise NotImplementedError(
                "Make sure to run __audio_seperator() before this function as self.audio_fids is not set."
            )
        with open("conf.yaml", "rb") as stream:
            conf = yaml.full_load(stream)
        conf_captions = conf["captions"]
        conf_data = conf["data"]
        conf_splits = [conf_data["splits"][split] for split in conf_data["splits"]]

        for split in conf_splits:
            cap_file = f"{conf_captions['caption_prefix']}{split}.csv"  # ? File name for current split
            captions = pd.read_csv(
                os.path.join(conf_captions["captions_dir"], cap_file)
            )

    # ? Seperate audio files
    def __call__(self):
        self.__caption_seperator()


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
            self.audio_params.sample_rate = sample_rate
            self.audio_params.window_length_secs = window_length_secs
            self.audio_params.hop_length_secs = hop_length_secs
            self.audio_params.num_mels = num_mels
            self.audio_params.fmin = fmin
            self.audio_params.fmax = fmax
            self.audio_params.log_offset = log_offset

        # Create a log_mel_spectogram
        def __log_mel_spectogram(self, y: List[float]):
            type_check = Typing_Utils()

            if not len(self.audio_params):
                raise NotImplementedError(
                    "audio_params not set, use set_audio_params()"
                )

            if not type_check.list_checker(y, [int, float]):
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

            return self.mel_spectrogram

        # Turn the audio into an h5py file
        def dir_to_h5py_logmel(self):
            # Load config
            with open("conf.yaml", "rb") as stream:
                conf = yaml.full_load(stream)
            conf_data = conf["data"]

            output_file = os.path.join(conf_data["dataset_dir"], conf_data["hdf5_file"])

            # Load splits
            conf_splits = [conf_data["splits"][split] for split in conf_data["splits"]]
            # with open(os.path.join(conf_data["dataset_dir"], "audio_info.pkl"), "rb") as store:

        #     global_params["audio_fids"] = pickle.load(store)["audio_fids"]
        # with h5py.File(output_file, "w") as feature_store:


if __name__ == "__main__":
    preprocess = Preprocess()
    preprocess()
