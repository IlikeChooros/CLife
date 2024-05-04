#include <core/CNN.hpp>

START_NAMESPACE_NEURAL_NETWORK


cnn::cnn(
  int input_channels,
  int input_size,
  int number_of_kernels,
  int kernel_size,
  int stride,
  int padding,
  int pooling_kernel_size,
  int pooling_stride,
  int pooling_padding
)
{
  (void)build(
    input_channels,
    input_size,
    number_of_kernels,
    kernel_size,
    stride,
    padding,
    pooling_kernel_size,
    pooling_stride,
    pooling_padding
  );
}

cnn& cnn::build(
  int input_channels,
  int input_size,
  int number_of_kernels,
  int kernel_size,
  int stride,
  int padding,
  int pooling_kernel_size,
  int pooling_stride,
  int pooling_padding
)
{
  (void)_conv.build(
    kernel_size,
    number_of_kernels,
    input_channels,
    stride,
    padding
  );
  (void)_pool.build(
    pooling_kernel_size,
    number_of_kernels,
    stride,
    padding
  );
  size_t output_size = _pool.get_output_size(_conv.get_output_size(input_size));

  (void)_FC.build(
    {output_size * output_size * number_of_kernels, 10},
    ActivationType::softmax, ActivationType::relu, 0.2
  );

  return *this;
}

void cnn::init()
{
  _conv.initialize();
  _FC.initialize();
}

void cnn::forward(matrix3d_t& input)
{
  _NetworkFeedData feed(_FC._output_layer, _FC._hidden_layers);
  feed_forward(input, feed);
}

void cnn::feed_forward(matrix3d_t& input, _NetworkFeedData& feed_data)
{
  input = _conv.forward(input);
  input = _pool.forward(input);
  
  auto flattened = flatten(input[0]);
  _FC.feed_forward(feed_data, flattened);
}

void cnn::backprop(const matrix3d_t& input, vector_t& target)
{
  auto ref = input;
  _NetworkFeedData feed(_FC._output_layer, _FC._hidden_layers);
  feed_forward(ref, feed);

  _FC.backprop(feed, target);
  
  auto pool_output_size = _pool.get_output_size(_conv.get_output_size(ref[0].size()));
  matrix3d_t prev_partial_dervis = reshape(
    feed._layer_feed_data[0]._partial_derivatives, 
    _pool._input_channels, 
    pool_output_size, 
    pool_output_size
  );
  
  prev_partial_dervis = _pool.backprop(prev_partial_dervis);
  _conv.backprop(prev_partial_dervis, ref);
}

void cnn::apply(double learning_rate, size_t batch_size)
{
  _FC.apply(learning_rate, batch_size);
  _conv.apply_gradients(learning_rate, batch_size);
}


double cnn::cost()
{
  return _FC.cost();
}

END_NAMESPACE