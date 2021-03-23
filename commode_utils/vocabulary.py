import pickle
from abc import abstractmethod
from collections import Counter
from os.path import join, exists, dirname
from typing import Dict, Counter as CounterType

from omegaconf import DictConfig
from tqdm.auto import tqdm

from commode_utils.filesystem import count_lines_in_file


class BaseVocabulary:
    _vocab_filename = "vocabulary.pkl"
    _log_filename = "bad_samples.log"

    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"

    _separator = "|"
    _label = "label"
    _token = "token"
    _node = "node"

    def __init__(self, config: DictConfig, data_folder: str):
        vocabulary_file = join(data_folder, config.name, self._vocab_filename)
        if not exists(vocabulary_file):
            raise ValueError(f"Can't find vocabulary file ({vocabulary_file})")
        with open(vocabulary_file, "rb") as f_in:
            self._counters: Dict[str, CounterType[str]] = pickle.load(f_in)

        self._label_to_id = {self.PAD: 0, self.UNK: 1, self.SOS: 2, self.EOS: 3}
        self._label_to_id.update(
            (token[0], i + 4) for i, token in enumerate(self._counters[self._label].most_common(config.max_labels))
        )

        self._token_to_id = {self.PAD: 0, self.UNK: 1, self.SOS: 2, self.EOS: 3}
        self._token_to_id.update(
            (token[0], i + 4) for i, token in enumerate(self._counters[self._token].most_common(config.max_tokens))
        )

        self._node_to_id = {self.PAD: 0, self.UNK: 1}
        self._node_to_id.update((token, i + 2) for i, token in enumerate(self._counters[self._node]))

    @property
    def label_to_id(self) -> Dict[str, int]:
        return self._label_to_id

    @property
    def token_to_id(self) -> Dict[str, int]:
        return self._token_to_id

    @property
    def node_to_id(self) -> Dict[str, int]:
        return self._node_to_id

    @staticmethod
    @abstractmethod
    def _process_raw_sample(raw_sample: str, counters: Dict[str, CounterType[str]]):
        raise NotImplementedError()

    @staticmethod
    def build_from_scratch(train_data: str):
        total_samples = count_lines_in_file(train_data)
        counters: Dict[str, CounterType[str]] = {
            key: Counter() for key in [BaseVocabulary._label, BaseVocabulary._token, BaseVocabulary._node]
        }
        with open(train_data, "r") as f_in:
            for raw_sample in tqdm(f_in, total=total_samples):
                BaseVocabulary._process_raw_sample(raw_sample, counters)

        for feature, counter in counters.items():
            print(f"Count {len(counter)} {feature}, top-5: {counter.most_common(5)}")

        dataset_dir = dirname(train_data)
        vocabulary_file = join(dataset_dir, BaseVocabulary._vocab_filename)
        with open(vocabulary_file, "wb") as f_out:
            pickle.dump(counters, f_out)
