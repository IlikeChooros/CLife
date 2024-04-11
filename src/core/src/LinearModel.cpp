#include <core/LinearModel.hpp>

START_NAMESPACE_NEURAL_NETWORK


LinearModel::LinearModel(std::size_t inputs): 
  _inputs(inputs), _output(0),
  _weights(inputs, real_number_t(0)), _bias(0),
  _partial_derviative(0), _gradient_weights(inputs, real_number_t(0)), 
  _gradient_bias(0)
{
  randomize(&_weights, inputs);
}

void LinearModel::apply(double learning_rate, size_t batch_size)
{
  auto size = _weights.size();
  for (std::size_t i = 0; i < size; ++i)
  {
    _weights[i] -= learning_rate * _gradient_weights[i] / batch_size;
    _gradient_weights[i] = 0;
  }
  _bias -= learning_rate * _gradient_bias / batch_size;
  _gradient_bias = 0;
}

void LinearModel::update_gradients()
{
  _gradient_bias += _partial_derviative;
  std::transform(_inputs.begin(), _inputs.end(), _gradient_weights.begin(), _gradient_weights.begin(), 
    [this](real_number_t input, real_number_t gradient_weight) -> real_number_t
    {
      return gradient_weight + input * _partial_derviative;
    });
}

void LinearModel::batch_learn(data::data_batch* batch, double learning_rate)
{
  for (auto& data : *batch)
  {
    set_inputs(data.input);
    _partial_derviative = (output() - data.expect[0]) * 2;
    update_gradients();
  }
  apply(learning_rate, batch->size());
}

real_number_t LinearModel::cost(real_number_t expected_output)
{
  auto diff = expected_output - _output;
  return diff * diff;
}

void LinearModel::set_inputs(const vector_t& inputs)
{
  _inputs = inputs;
}

real_number_t LinearModel::output()
{
  return std::inner_product(_inputs.begin(), _inputs.end(), _weights.begin(), _bias);
}

END_NAMESPACE