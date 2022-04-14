from typing import List


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

        def log_mel_spectogram(self, y: List[float]):
            if not len(self.audio_params):
                raise NotImplementedError(
                    "audio_params not set, use set_audio_params()"
                )

            if not isinstance(y, List[float]):
                raise TypeError("y must be a list of floats")
            return self.audio_params


if __name__ == "__main__":
    utils = Utils()
    audio = utils.Audio()
    print(audio.log_mel_spectogram())
