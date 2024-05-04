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
  for(auto& kernel : _weights)
  {
    for(auto& weights : kernel)
    {
      randomize(&weights, _kernel_size);
    }
  }

  // _weights[0] = {
  //   // {0.25, 0, -0.25},
  //   // {0.5, 0, -0.5},
  //   // {0.25, 0, -0.25}

  //   // {0.25, 0.5, 0.25},
  //   // {0, 0, 0},
  //   // {-0.25, -0.5, -0.25}

  //   {0, 0, 0},
  //   {0, 1, 0},
  //   {0, 0, 0},

  // };

  //  Gaussian kernel
  // for (int x = -1; x < 2; x++){
  //       for (int y = -1; y < 2; y++){
  //           _weights[0][x+1][y+1] = exp(-(x*x + y*y) / 2.0) / (2 * M_PI);
  //       }
  //   }
}

matrix3d_t ConvLayer::forward(const matrix3d_t& input)
{
  int input_size = input[_input_channels - 1].size();
  int output_size = get_output_size(input_size);
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
                sum += ReLU::activation(input[channel][x][y] * _weights[kernel][k][l]);
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

void ConvLayer::backprop(matrix3d_t& partial_dervis, matrix3d_t& inputs)
{
  int input_size = inputs[_input_channels - 1].size();

  for(int kernel = 0; kernel < _number_of_kernels; ++kernel)
  {
    for(int i = 0; i < _kernel_size; ++i)
    {
      for(int j = 0; j < _kernel_size; ++j)
      {
        double sum = 0.0;
        for(int channel = 0; channel < _input_channels; ++channel)
        {
          for(int k = 0; k < input_size; ++k)
          {
            for(int l = 0; l < input_size; ++l)
            {
              int x = k - i + _padding;
              int y = l - j + _padding;
              if(x >= 0 && x < input_size && y >= 0 && y < input_size)
              {
                sum += inputs[channel][k][l] * partial_dervis[kernel][x][y];
              }
            }
          }
        }
        _gradient_weights[kernel][i][j] = sum;
      }
    }
  }
}

void ConvLayer::apply_gradients(double learn_rate, size_t batch_size)
{
  for(int kernel = 0; kernel < _number_of_kernels; ++kernel)
  {
    for(int i = 0; i < _kernel_size; ++i)
    {
      for(int j = 0; j < _kernel_size; ++j)
      {
        _weights[kernel][i][j] -= learn_rate * _gradient_weights[kernel][i][j] / batch_size;
      }
    }
  }
}

END_NAMESPACE