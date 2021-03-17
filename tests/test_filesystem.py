from unittest import TestCase

from commode_utils.filesystem import count_lines_in_file, get_lines_offsets, get_line_by_offset
from tests.utils import ZEN


class TestFilesystemUtils(TestCase):

    # True offsets are obtained from this command:
    # grep -bo ".*$"  tests/resources/zen.txt
    _zen_true_offsets = [0, 31, 65, 96, 132, 160, 189, 209, 265, 301, 336, 364, 422, 492, 559, 585, 634, 693, 758]
    _zen_n_lines = len(_zen_true_offsets)

    def test_count_lines_in_file(self):
        count_n_lines = count_lines_in_file(ZEN)
        self.assertEqual(self._zen_n_lines, count_n_lines)

    def test_get_lines_offsets(self):
        count_offsets = get_lines_offsets(ZEN)
        self.assertListEqual(self._zen_true_offsets, count_offsets)

    def test_get_line_by_offset(self):
        with open(ZEN, "r") as zen_file:
            for line, offset in zip(zen_file, self._zen_true_offsets):
                offset_line = get_line_by_offset(ZEN, offset)
                line = line.strip()
                self.assertSequenceEqual(line, offset_line)
