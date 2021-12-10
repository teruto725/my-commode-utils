import unittest
from collections import Counter

from parameterized import parameterized

from commode_utils.vocabulary import BaseVocabulary


class TestVocabulary(unittest.TestCase):
    @parameterized.expand([[None, ["a", "b", "c"]], [1, ["a", "b", "c"]], [2, ["a", "b"]], [3, ["a"]]])
    def test_extract_tokens(self, border, result):
        counter = Counter(["a", "a", "a", "b", "b", "c"])
        extracted_tokens = BaseVocabulary._extract_tokens_by_count(counter, border)
        self.assertListEqual(result, extracted_tokens)
