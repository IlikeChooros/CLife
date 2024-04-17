#pragma once


#include "namespaces.hpp"
#include "types.hpp"

START_NAMESPACE_NEURAL_NETWORK

class ConvLayer{

  int _kernel_size;
  int _stride;
  int _padding;
  int _input_channels;
  int _output_channels;
  
  matrix_t _weights;

  public:
  ConvLayer() = default;
  ConvLayer(int kernel_size, int stride, int padding, int input_channels, int output_channels);
  ConvLayer& build(int kernel_size, int stride, int padding, int input_channels, int output_channels);
  void initialize();

  vector_t forward(const vector_t& input);
};

END_NAMESPACE