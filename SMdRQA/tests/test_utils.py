import unittest
import numpy as np
from SMdRQA.utils import *

# The assert_matrix function

# Test case class for assert_matrix


class TestAssertMatrix(unittest.TestCase):

    def test_1d_array(self):
        # Test with a 1D array, which should be converted to a 2D array
        input_array = np.array([1, 2, 3])
        result = assert_matrix(input_array)
        self.assertEqual(result.shape, (1, 3))
        np.testing.assert_array_equal(result, np.array([[1, 2, 3]]))

    def test_2d_array(self):
        # Test with a 2D array, which should be returned as is
        input_array = np.array([[1, 2, 3], [4, 5, 6]])
        result = assert_matrix(input_array)
        self.assertEqual(result.shape, (2, 3))
        np.testing.assert_array_equal(result, input_array)

    def test_3d_array(self):
        # Test with a 3D array, which should be returned as is
        input_array = np.array([[[1, 2, 3], [4, 5, 6]]])
        result = assert_matrix(input_array)
        self.assertEqual(result.shape, (1, 2, 3))
        np.testing.assert_array_equal(result, input_array)

    def test_non_numpy_input(self):
        # Test with a non-NumPy array input, which should raise a ValueError
        input_list = [1, 2, 3]
        with self.assertRaises(ValueError):
            assert_matrix(input_list)


class TestCompute3DMatrixSize(unittest.TestCase):

    def test_random_float64_matrix(self):
        # Define matrix dimensions
        dim1, dim2, dim3 = 100, 200, 300

        # Create a random matrix of the given dimensions and dtype float64
        matrix = np.random.rand(dim1, dim2, dim3).astype(np.float64)

        # Compute the expected size in GiB using numpy's nbytes attribute
        expected_size = matrix.nbytes / (1024**3)

        # Compute the size using the function
        result = compute_3D_matrix_size(dim1, dim2, dim3, dtype=np.float64)

        # Assert the two sizes are almost equal (due to floating-point
        # precision)
        self.assertAlmostEqual(result, expected_size, places=6)

    def test_random_float32_matrix(self):
        # Define matrix dimensions
        dim1, dim2, dim3 = 50, 60, 70

        # Create a random matrix of the given dimensions and dtype float32
        matrix = np.random.rand(dim1, dim2, dim3).astype(np.float32)

        # Compute the expected size in GiB using numpy's nbytes attribute
        expected_size = matrix.nbytes / (1024**3)

        # Compute the size using the function
        result = compute_3D_matrix_size(dim1, dim2, dim3, dtype=np.float32)

        # Assert the two sizes are almost equal (due to floating-point
        # precision)
        self.assertAlmostEqual(result, expected_size, places=6)


class TestAssert3DMatrixSize(unittest.TestCase):

    def test_memory_within_limit(self):
        # Test case where the memory requirement should be within the specified
        # limit
        dim1, dim2, dim3 = 100, 100, 100  # Adjust dimensions to be within the 4 GiB limit
        memory_limit = 4  # GiB
        self.assertTrue(
            assert_3D_matrix_size(
                dim1,
                dim2,
                dim3,
                dtype=np.float64,
                memory_limit=memory_limit))

    def test_memory_exceeds_limit(self):
        # Test case where the memory requirement should exceed the specified
        # limit
        dim1, dim2, dim3 = 1000, 1000, 1000  # Dimensions likely to exceed 4 GiB
        memory_limit = 4  # GiB
        self.assertFalse(
            assert_3D_matrix_size(
                dim1,
                dim2,
                dim3,
                dtype=np.float64,
                memory_limit=memory_limit))

    def test_different_dtype_within_limit(self):
        # Test case with a different data type, e.g., float32, which uses less
        # memory
        dim1, dim2, dim3 = 500, 500, 500
        memory_limit = 2  # GiB
        self.assertTrue(
            assert_3D_matrix_size(
                dim1,
                dim2,
                dim3,
                dtype=np.float32,
                memory_limit=memory_limit))

    def test_edge_case_minimal_matrix(self):
        # Edge case with a very small matrix
        dim1, dim2, dim3 = 1, 1, 1
        memory_limit = 0.000001  # GiB
        self.assertTrue(
            assert_3D_matrix_size(
                dim1,
                dim2,
                dim3,
                dtype=np.float64,
                memory_limit=memory_limit))

    def test_large_memory_limit(self):
        # Test case with a large memory limit where any reasonable matrix size
        # would pass
        dim1, dim2, dim3 = 100, 100, 100
        memory_limit = 1000  # GiB
        self.assertTrue(
            assert_3D_matrix_size(
                dim1,
                dim2,
                dim3,
                dtype=np.float64,
                memory_limit=memory_limit))
