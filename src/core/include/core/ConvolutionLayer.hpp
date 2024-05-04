#pragma once


#include "activation.hpp"
#include "utils.hpp"
#include "types.hpp"

START_NAMESPACE_NEURAL_NETWORK

/**
 * @brief Convolutional layer with max pooling
*/
class ConvLayer{
  public:

  struct FeedData{
    FeedData(ConvLayer& layer, int input_size){
      int _output_size = layer.get_output_size(input_size);
      _outputs.resize(layer._number_of_kernels, matrix_t(_output_size, vector_t(_output_size)));
      _outputs.resize(layer._number_of_kernels, matrix_t(_output_size, vector_t(_output_size)));
      _partial_dervis.resize(layer._number_of_kernels, matrix_t(_output_size, vector_t(_output_size)));
    }

    matrix3d_t _outputs;
    matrix3d_t _weighted_sum;
    matrix3d_t _partial_dervis;
  };

  ConvLayer(
    int kernel_size = 3, 
    int number_of_kernels = 1, 
    int input_channels = 1,  
    int stride = 1, 
    int padding = 1
  );
  
  /**
   * @brief Builds the convolutional layer
   * @param kernel_size the size of the convolutional matrix (filter, kernel)
   * @param number_of_kernels the number of filters
   * @param input_channels the number of channels (RGB = 3, Grayscale = 1, etc.)
   * @param stride the number of pixels the filter moves in each step
   * @param padding basically this is used to make the output size the same as the input size, = 0 makes the output smaller by kernel_size - 1
   * @return reference to this object
  */
  ConvLayer& build(
    int kernel_size = 3, 
    int number_of_kernels = 1, 
    int input_channels = 1, 
    int stride = 1, 
    int padding = 1
  );

  /**
   * @brief Initializes the weights with random values
  */
  void initialize();

  /**
   * @brief Forward pass of the convolutional layer
   * @param input the input matrix
   * @return the output matrix - size: (number_of_kernels, output_size, output_size), output_size = (input_size - kernel_size + 2 * padding) / stride + 1
  */
  matrix3d_t forward(const matrix3d_t& input);

  /**
   * @brief Backward pass of the convolutional layer
   * @param partial_dervis the partial derivatives
  */
  void backprop(matrix3d_t& partial_dervis, matrix3d_t& inputs);
  
  void apply_gradients(double learn_rate, size_t batch_size);

  /**
   * @brief Returns the output of the convolutional layer
   * @return the output of the convolutional layer
  */
  inline size_t get_output_size(size_t input_size){
    return (input_size - _kernel_size + 2 * _padding) / _stride + 1;
  }

  // Also known as filter, the size of the convolutional matrix
  int _kernel_size;
  // The number of filters, creates the depth of the output (3rd dimension of the output matrix)
  int _number_of_kernels;
  // The number of pixels the filter moves in each step
  int _stride;
  // Basically this is used to make the output size the same as the input size
  int _padding;
  // The number of channels (RGB = 3, Grayscale = 1, etc.)
  int _input_channels;

  matrix3d_t _weights;
  matrix3d_t _gradient_weights;
};

END_NAMESPACE