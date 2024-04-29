#pragma once

#include <numeric>
#include <limits>

#include "types.hpp"

START_NAMESPACE_NEURAL_NETWORK

class MaxPoolingLayer{
  int _kernel_size;
  int _stride;
  int _padding;
  int _input_channels;

  // Used to store the indexes of the max values in the input matrix
  matrix_t _max_indexes;

  public:
  MaxPoolingLayer(
    int kernel_size = 2, 
    int input_channels = 1, 
    int stride = 2, 
    int padding = 0
  );

  MaxPoolingLayer& build(
    int kernel_size = 2, 
    int input_channels = 1, 
    int stride = 2, 
    int padding = 0
  );

  /**
   * @brief Forward pass of the max pooling layer
   * @param input the input matrix
   * @return the output matrix, size = (input_channels, output_size, output_size), output_size = (input_size - kernel_size + 2 * padding) / stride + 1
  */
  matrix3d_t forward(const matrix3d_t& input);

  

  /**
   * @brief Returns the indexes of the max values in the input matrix
   * @return the indexes of the max values in the input matrix, stored in format: [polling_index][0, 1] -> x, y position
  */
  const matrix_t& get_max_indexes();
};


END_NAMESPACE