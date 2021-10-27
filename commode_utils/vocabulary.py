import pickle
from abc import abstractmethod
from collections import Counter
from os.path import join, exists, dirname
from typing import Dict, Counter as TCounter, Type, Optional, List

from tqdm.auto import tqdm

from commode_utils.filesystem import count_lines_in_file


class BaseVocabulary:
    vocab_filename = "vocabulary.pkl"
    _log_filename = "bad_samples.log"

    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"

    _separator = "|"
    LABEL = "label"
    TOKEN = "token"
    NODE = "node"

    def __init__(self, vocabulary_file: str, labels_count: Optional[int] = None, tokens_count: Optional[int] = None):
        if not exists(vocabulary_file):
            raise ValueError(f"Can't find vocabulary file ({vocabulary_file})")
        with open(vocabulary_file, "rb") as f_in:
            self._counters: Dict[str, TCounter[str]] = pickle.load(f_in)

        self._label_to_id = {self.PAD: 0, self.UNK: 1, self.SOS: 2, self.EOS: 3}
        labels = self._extract_tokens_by_count(self._counters[self.LABEL], labels_count)
        self._label_to_id.update((token, i + 4) for i, token in enumerate(labels))

        self._token_to_id = {self.PAD: 0, self.UNK: 1, self.SOS: 2, self.EOS: 3}
        tokens = self._extract_tokens_by_count(self._counters[self.TOKEN], tokens_count)
        self._token_to_id.update((token, i + 4) for i, token in enumerate(tokens))

        self._node_to_id = {self.PAD: 0, self.UNK: 1}
        self._node_to_id.update((token, i + 2) for i, token in enumerate(self._counters[self.NODE]))

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
    def process_raw_sample(raw_sample: str, counters: Dict[str, TCounter[str]]):
        raise NotImplementedError()

    @staticmethod
    def _extract_tokens_by_count(counter: TCounter, count_border: Optional[int]) -> List[str]:
        words, counts = zip(*counter.most_common())
        if count_border is None:
            return words
        border = [i for i, c in enumerate(counts) if c < count_border][0]
        return words[:border]


def build_from_scratch(train_data: str, vocabulary_cls: Type[BaseVocabulary]):
    total_samples = count_lines_in_file(train_data)
    counters: Dict[str, TCounter[str]] = {
        key: Counter() for key in [vocabulary_cls.LABEL, vocabulary_cls.TOKEN, vocabulary_cls.NODE]
    }
    with open(train_data, "r") as f_in:
        for raw_sample in tqdm(f_in, total=total_samples):
            vocabulary_cls.process_raw_sample(raw_sample, counters)

    for feature, counter in counters.items():
        print(f"Count {len(counter)} {feature}, top-5: {counter.most_common(5)}")

    dataset_dir = dirname(train_data)
    vocabulary_file = join(dataset_dir, vocabulary_cls.vocab_filename)
    with open(vocabulary_file, "wb") as f_out:
        pickle.dump(counters, f_out)
