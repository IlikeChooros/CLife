#include <core/MaxPool.hpp>

START_NAMESPACE_NEURAL_NETWORK

MaxPoolingLayer::MaxPoolingLayer(int kernel_size, int input_channels, int stride, int padding)
{
  (void)build(kernel_size, input_channels, stride, padding);
}

MaxPoolingLayer& MaxPoolingLayer::build(int kernel_size, int input_channels, int stride, int padding)
{
  _kernel_size = kernel_size;
  _input_channels = input_channels;
  _stride = stride;
  _padding = padding;

  return *this;
}

matrix3d_t MaxPoolingLayer::forward(const matrix3d_t& input)
{
  _input_size = input[_input_channels - 1].size();
  int output_size = get_output_size(_input_size);
  matrix3d_t output(_input_channels, matrix_t(output_size, vector_t(output_size, 0.0)));
  _max_indexes = matrix3d_t(output_size, matrix_t(output_size, vector_t(2, 0))); // 2 -> x, y position

  for(int channel = 0; channel < _input_channels; ++channel)
  {
    for(int i = 0; i < output_size; ++i)
    {
      for(int j = 0; j < output_size; ++j)
      {
        real_number_t max_value = -std::numeric_limits<real_number_t>::max();
        for(int k = 0; k < _kernel_size; ++k)
        {
          for(int l = 0; l < _kernel_size; ++l)
          {
            int x = i * _stride + k - _padding;
            int y = j * _stride + l - _padding;
            if(x >= 0 && x < _input_size && y >= 0 && y < _input_size)
            {
              if (input[channel][x][y] > max_value)
              {
                max_value = input[channel][x][y];
                _max_indexes[i][j][0] = x;
                _max_indexes[i][j][1] = y;
              }
            }
          }
        }
        output[channel][i][j] = max_value;
      }
    }
  }

  return output;
}

matrix3d_t MaxPoolingLayer::backprop(matrix3d_t& partial_dervis)
{
  int output_size = partial_dervis[0].size();
  matrix3d_t input(_input_channels, matrix_t(_input_size, vector_t(_input_size, 0.0)));

  for(int channel = 0; channel < _input_channels; ++channel)
  {
    for(int i = 0; i < output_size; ++i)
    {
      for(int j = 0; j < output_size; ++j)
      {
        input[channel][_max_indexes[i][j][0]][_max_indexes[i][j][1]] = partial_dervis[channel][i][j];
      }
    }
  }
  
  return input;
}

const matrix3d_t& MaxPoolingLayer::get_max_indexes()
{
  return _max_indexes;
}

END_NAMESPACE