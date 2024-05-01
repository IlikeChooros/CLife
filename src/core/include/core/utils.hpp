#pragma once

#include <chrono>
#include <random>
#include <algorithm>

#include "namespaces.hpp"
#include "types.hpp"

START_NAMESPACE_NEURAL_NETWORK

/**
 * @brief Randomizes the given vector, with values between -1/sqrt(input_size) and 1/sqrt(input_size)
 * @param vec vector to randomize
 * @param input_size size of the input
*/
void randomize(vector_t* vec, std::size_t input_size);

/**
 * @brief Flattens the given matrix into a vector
 * @param matrix matrix to flatten
 * @return flattened vector
*/
vector_t flatten(const matrix_t& matrix);

/**
 * @brief Flattens the given 3D matrix into a vector
 * @param matrix matrix to flatten
 * @return flattened vector
*/
vector_t flatten(const matrix3d_t& matrix);

/**
 * @brief Reshapes the given vector into a matrix with the given number of rows and columns
 * @param vec vector to reshape
 * @param rows number of rows
 * @param cols number of columns
 * @return reshaped matrix (rows, cols)
*/
matrix_t reshape(const vector_t& vec, std::size_t rows, std::size_t cols);

/**
 * @brief Reshapes the given vector into a 3D matrix with the given number of channels, rows and columns
 * @param vec vector to reshape
 * @param channels number of channels
 * @param rows number of rows
 * @param cols number of columns
 * @return reshaped 3D matrix (channels, rows, cols)
*/
matrix3d_t reshape(const vector_t& vec, std::size_t channels, std::size_t rows, std::size_t cols);

END_NAMESPACE