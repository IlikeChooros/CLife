#pragma once

#include <stddef.h>
#include <numeric>
#include <limits>

#include "types.hpp"


START_NAMESPACE_NEURAL_NETWORK

class MaxPoolingLayer{
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
   * @brief Backward pass of the max pooling layer
   * @param output the output matrix
   * @return the backward pass matrix, creates a matrix 
  */
  matrix3d_t backprop(matrix3d_t& partial_dervis);

  
  /**
   * @brief Returns the indexes of the max values in the input matrix
   * @return the indexes of the max values in the input matrix, stored in format: [polling_index][0, 1] -> x, y position
  */
  const matrix3d_t& get_max_indexes();

  /**
   * @brief Returns the output of the convolutional layer
   * @return the output of the convolutional layer
  */
  inline size_t get_output_size(size_t input_size){
    return (input_size - _kernel_size + 2 * _padding) / _stride + 1;
  }

  int _kernel_size;
  int _stride;
  int _padding;
  int _input_channels;
  int _input_size;

  // Used to store the indexes of the max values in the input matrix
  matrix3d_t _max_indexes;
  matrix3d_t _output;
};


END_NAMESPACE