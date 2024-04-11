#include <core/utils.hpp>

START_NAMESPACE_NEURAL_NETWORK

void randomize(vector_t* vec, std::size_t input_size)
{
  std::default_random_engine _engine(std::chrono::system_clock::now().time_since_epoch().count());
  double limit = sqrt(1.0 / input_size);
  std::uniform_real_distribution<double> _dist(-limit, limit);

  std::generate(vec->data(), vec->data() + vec->size(), [&_dist, &_engine](){ return _dist(_engine); });
}

END_NAMESPACE