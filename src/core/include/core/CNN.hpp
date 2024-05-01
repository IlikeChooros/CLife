#pragma once

#include "ONeural.hpp"
#include "ConvolutionLayer.hpp"
#include "MaxPool.hpp"

START_NAMESPACE_NEURAL_NETWORK


class cnn{

  ONeural _FC;
  ConvLayer _conv;
  MaxPoolingLayer _pool;

  public:
  cnn(
    int input_channels = 1,
    int input_size = 28,
    int number_of_kernels = 1,
    int kernel_size = 3,
    int stride = 1,
    int padding = 0,
    int pooling_kernel_size = 2,
    int pooling_stride = 2,
    int pooling_padding = 0
  );

  cnn& build(
    int input_channels = 1,
    int input_size = 28,
    int number_of_kernels = 1,
    int kernel_size = 3,
    int stride = 1,
    int padding = 0,
    int pooling_kernel_size = 2,
    int pooling_stride = 2,
    int pooling_padding = 0
  );

  /**
   * @brief Initializes the network with random weights
  */
  void init();

  /**
   * @brief Forward pass of the network
   * @param input the input matrix
  */
  void forward(matrix3d_t& input);

  /**
   * @brief Feed forward of the network
   * @param input the input matrix
   * @param feed_data the feed data
  */
  void feed_forward(matrix3d_t& input, _NetworkFeedData& feed_data);

  /**
   * @brief Backpropagation of the network
   * @param input the input matrix
   * @param target the target vector (expected output of the network)
  */
  void backprop(matrix3d_t& input, vector_t& target);

  /**
   * @brief Applies the gradients to the weights
   * @param learn_rate the learning rate
  */
  void apply_gradients(double learn_rate);

  /**
   * @brief Returns the output of the fully connected layer
   * @return the output of the fully connected layer
  */
  const vector_t& outputs();
};

END_NAMESPACE