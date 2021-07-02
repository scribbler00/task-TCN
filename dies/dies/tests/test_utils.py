import numpy as np
import pandas as pd
import unittest
from numpy.testing import assert_almost_equal, assert_array_less, assert_array_equal
from dies.utils import get_structure


class TestGetStructure(unittest.TestCase):
    def test_correct_final_output_for_final_outputs(self):
        structure = get_structure(100, 10, 5, final_outputs=[10, 1])

        self.assertEqual(structure[-1], 1)
        self.assertEqual(structure[-2], 10)

        # none of final_outputs is converted to empty list
        structure = get_structure(10, 50, 5, final_outputs=None)
        self.assertEqual(structure[-1], 5)

        self.assertRaises(ValueError, get_structure, 100, 10, 5, None, 0)
        self.assertRaises(ValueError, get_structure, 100, 10, 5, None, [1, 0])
        self.assertRaises(ValueError, get_structure, 100, 10, 5, None, [1, None])

    def test_percentual_conversion(self):
        structure_1 = get_structure(100, 10, 5, final_outputs=None)
        structure_2 = get_structure(100, 0.1, 5, final_outputs=None)

        self.assertCountEqual(structure_1, structure_2)

    def test_reducage(self):
        structure = get_structure(10, 1, 5, final_outputs=None)

        self.assertCountEqual(structure, [10, 9, 8, 7, 6, 5])
