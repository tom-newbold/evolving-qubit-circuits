import unittest
from linear_genetic_programming_utils import *

class test_lgp_utils(unittest.TestCase):
    def test_encode_to_letter(self):
        self.assertEqual(encode_to_letter(0),'A')
        self.assertEqual(encode_to_letter(25),'Z')
        self.assertEqual(encode_to_letter(30),'e')
        self.assertEqual(encode_to_letter(50),'y')

        self.assertEqual(encode_to_letter(55),'Λ')
        self.assertEqual(encode_to_letter(65),'δ')
        self.assertEqual(encode_to_letter(75),'τ')

        # out of range values return None
        self.assertIsNone(encode_to_letter(85))
        self.assertIsNone(encode_to_letter(-5))

    def test_basis_states(self):
        states = basis_states()
        self.assertIn(Statevector.from_int(0, 8), states)
        self.assertIn(Statevector.from_int(2, 8), states)
        self.assertIn(Statevector.from_int(7, 8), states)
        # states for different number of qubits not present
        self.assertNotIn(Statevector.from_int(0, 2**5), states)
        self.assertNotIn(Statevector.from_int(17, 2**5), states)

        states = basis_states(5)
        self.assertIn(Statevector.from_int(1, 2**5), states)
        self.assertIn(Statevector.from_int(8, 2**5), states)
        self.assertIn(Statevector.from_int(17, 2**5), states)

    def test_list_avr(self):
        with self.assertRaises(ZeroDivisionError):
            # average of an empty list is undefined
            list_avr([])
        for x in range(5,10):
            x = float(x)
            self.assertEqual(list_avr([x]),x)
            self.assertEqual(list_avr([x,x,x,x,x]),x)
            # for evenly distributed ordered list, average is equal to median
            self.assertEqual(list_avr([x,2*x,3*x,4*x,5*x]),3*x)

    def test_get_averages_list(self):
        self.assertIsNone(get_averages_list(0))
        self.assertIsNone(get_averages_list(' test__ '))
        
        lists = [list(range(5*i, 5*(i+1))) for i in range(10)]
        base_sum = sum([5*i for i in range(10)])
        average_list = [(base_sum + 10*i)/10 for i in range(5)]
        self.assertEqual(get_averages_list(lists),average_list)

    def test_get_max_list(self):
        self.assertIsNone(get_max_list(0))
        self.assertIsNone(get_max_list(' test__ '))

        lists = [list(range(5*i, 5*(i+1))) for i in range(10)]
        self.assertEqual(get_max_list(lists),list(range(45, 50)))

    def test_get_min_list(self):
        self.assertIsNone(get_min_list(0))
        self.assertIsNone(get_min_list(' test__ '))

        lists = [list(range(5*i, 5*(i+1))) for i in range(10)]
        self.assertEqual(get_min_list(lists),list(range(5)))

    def test_smooth_line(self):
        self.assertIsNone(smooth_line(0))
        self.assertIsNone(smooth_line(' test__ '))

        test_list = [float(x) for x in range(11)]
        # no smoothing
        self.assertEqual(smooth_line(test_list, 20), test_list)
        self.assertEqual(smooth_line(test_list, 10), test_list)
        # smoothing
        test_list_smoothed = test_list.copy()
        test_list_smoothed[0] = 0.5
        test_list_smoothed[-1] = 9.5
        self.assertEqual(smooth_line(test_list, 0), test_list_smoothed)
        test_list_smoothed[0] = 1.0
        test_list_smoothed[1] = 1.5
        test_list_smoothed[-2] = 8.5
        test_list_smoothed[-1] = 9.0
        self.assertEqual(smooth_line(test_list), test_list_smoothed)
        
        test_list[5] = 0.0
        test_list[6] = 0.0
        self.assertEqual(smooth_line(test_list, 15),test_list)
        # smoothing applied
        self.assertNotEqual(smooth_line(test_list, 5),test_list)
        self.assertNotEqual(smooth_line(test_list),test_list)

    # == TEST PLOTTING FUNCS ?? ==
    # == TESTING remove_duplicates REQUIRES LIST OF GENOTYPES