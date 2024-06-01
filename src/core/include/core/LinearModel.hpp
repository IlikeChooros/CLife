#pragma once

#include "namespaces.hpp"
#include "utils.hpp"
#include <data/data.hpp>
#include <algorithm>
#include <numeric>

START_NAMESPACE_NEURAL_NETWORK

class LinearModel
{
  vector_t _inputs;
  real_number_t _output;
  vector_t _weights;
  real_number_t _bias;

  real_number_t _partial_derviative;
  vector_t _gradient_weights;
  real_number_t _gradient_bias;

public:
  LinearModel(std::size_t inputs);

  /**
   * @brief Applies calculated gradients to the weights and biases
   * @param learning_rate learning rate
   */
  void apply(double learning_rate, size_t batch_size);

  /**
   * @brief Updates the gradients for the weights and biases
   */
  void update_gradients();

  /**
   * @brief Calculates the gradients for the weights and biases
   * @param batch learning data
   * @param learning_rate learning rate
   */
  void batch_learn(data::data_batch *batch, double learning_rate);

  /**
   * @brief Sets the inputs for the model
   * @param inputs input values
   */
  void set_inputs(const vector_t &inputs);

  /**
   * @brief Calculates the cost of the model
   */
  real_number_t cost(real_number_t expected_output);

  /**
   * @brief Calculates the output of the model, dot product of the inputs and weights
   * @return the output of the model
   */
  real_number_t output();
};

END_NAMESPACE