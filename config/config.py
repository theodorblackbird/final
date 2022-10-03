import yaml
from abc import ABC, abstractmethod
from copy import copy, deepcopy



class BaseConfig(ABC):
    def __init__(self, path=None, raise_for_required=False, raise_for_unmatched_type=False):
        self.raise_for_required = raise_for_required
        self.raise_for_unmatched_type = raise_for_unmatched_type
        self.conf = {}
        with open(path, 'r') as f:
            yaml_conf = yaml.safe_load(f)
        self.conf = yaml_conf
        leaves = self.get_leaves(self.required_conf)
        for leaf in leaves :
            self.check_entry(leaf)
    def __getitem__(self, x):
        return self.conf[x]
    def __setitem__(self, x, y):
        pass

    @abstractmethod
    def required_conf(self) -> dict:
        ...
    @staticmethod
    def get_leaves(d):
        return list(BaseConfig._get_leaves(d, []))
    @staticmethod
    def _get_leaves(d, past):
        for k, v in d.items():
            if isinstance(v, dict):
                past_ = copy(past)
                past_.append(k)
                yield from BaseConfig._get_leaves(v, past_)
            else:
                past_ = copy(past)
                past_.append(k)
                yield past_

    def check_entry(self, entry):
        conf = deepcopy(self.conf)
        expected_type = deepcopy(self.required_conf)
        for k in entry :
            try:
                conf = conf[k]
            except KeyError:
                print(f"Can't find entry {entry} at level {k}")
                if self.raise_for_required:
                    raise
                return
            expected_type = expected_type[k]
        if expected_type != type(conf) :
            print(f"Type for entry {entry} is {type(conf)} but expected type is {expected_type}")
            if self.raise_for_unmatched_type :
                raise



class Tacotron2Config(BaseConfig):
    def __init__(self, path=None, raise_for_required=False, raise_for_unmatched_type=False):
        super().__init__(path, raise_for_required, raise_for_unmatched_type)

    required_conf = {
            "encoder" : {
                "conv_layer" : {
                    "filter" : int,
                    "kernel_size" : list
                    },
                "char_embedding_size" : int
                },
            }

class TrainConfig(BaseConfig):
    def __init__(self, path=None, raise_for_required=False, raise_for_unmatched_type=False):
        super().__init__(path, raise_for_required, raise_for_unmatched_type)
        required_conf = {
                "train" : {
                    "batch_size" : int,
                    "epochs": int,
                    "drop_remainder": bool,
                    "clip_norm": float
                    },
                "data": {
                    "transcript_path": str,
                    "audio_dir": str,
                    "checkpoint_path": str,
                    }
                }
if __name__ == "__main__":
    tac2conf = Tacotron2Config("/home/theodor/ircam_tts/config/configs/tacotron2.yaml")
    print(tac2conf["encoder"]["char_embedding_size"])
