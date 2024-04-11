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

END_NAMESPACE