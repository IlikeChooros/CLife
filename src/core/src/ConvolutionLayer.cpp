#include <core/ConvolutionLayer.hpp>

START_NAMESPACE_NEURAL_NETWORK

ConvLayer::ConvLayer(int kernel_size, int number_of_kernels, int input_channels, int stride, int padding)
{
  (void)build(kernel_size, number_of_kernels, input_channels, stride, padding);
}

ConvLayer& ConvLayer::build(int kernel_size, int number_of_kernels, int input_channels, int stride, int padding)
{
  _kernel_size = kernel_size;
  _number_of_kernels = number_of_kernels;
  _input_channels = input_channels;
  _stride = stride;
  _padding = padding;

  _weights.resize(number_of_kernels, matrix_t(kernel_size, vector_t(kernel_size)));
  _gradient_weights.resize(number_of_kernels, matrix_t(kernel_size, vector_t(kernel_size)));

  return *this;
}

void ConvLayer::initialize()
{
  // for(auto& kernel : _weights)
  // {
  //   for(auto& weights : kernel)
  //   {
  //     randomize(&weights, _kernel_size);
  //   }
  // }

  _weights[0] = {
    // {0.25, 0, -0.25},
    // {0.5, 0, -0.5},
    // {0.25, 0, -0.25}
    {0.25, 0.5, 0.25},
    {0, 0, 0},
    {-0.25, -0.5, -0.25}
  };
}

matrix3d_t ConvLayer::forward(const matrix3d_t& input)
{
  int input_size = input[_input_channels - 1].size();
  int output_size = (input_size - _kernel_size + 2 * _padding) / _stride + 1;
  matrix3d_t output(_number_of_kernels, matrix_t(output_size, vector_t(output_size, 0.0)));

  for(int kernel = 0; kernel < _number_of_kernels; ++kernel)
  {
    for(int i = 0; i < output_size; ++i)
    {
      for(int j = 0; j < output_size; ++j)
      {
        double sum = 0.0;
        for(int channel = 0; channel < _input_channels; ++channel)
        {
          for(int k = 0; k < _kernel_size; ++k)
          {
            for(int l = 0; l < _kernel_size; ++l)
            {
              int x = i * _stride - _padding + k;
              int y = j * _stride - _padding + l;
              if(x >= 0 && x < input_size && y >= 0 && y < input_size)
              {
                sum += input[channel][x][y] * _weights[kernel][k][l];
              }
            }
          }
        }
        output[kernel][i][j] = sum;
      }
    }
  }
  return output;
}

END_NAMESPACE