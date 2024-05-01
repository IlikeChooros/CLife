#include <core/utils.hpp>

START_NAMESPACE_NEURAL_NETWORK

void randomize(vector_t* vec, std::size_t input_size)
{
  std::default_random_engine _engine(std::chrono::system_clock::now().time_since_epoch().count());
  double limit = sqrt(1.0 / input_size);
  std::uniform_real_distribution<double> _dist(-limit, limit);

  std::generate(vec->data(), vec->data() + vec->size(), [&_dist, &_engine](){ return _dist(_engine); });
}

vector_t flatten(const matrix_t& matrix)
{
  vector_t vec(matrix.size() * matrix[0].size());
  for(std::size_t i = 0; i < matrix.size(); ++i)
  {
    std::copy(matrix[i].begin(), matrix[i].end(), vec.begin() + i * matrix[i].size());
  }
  return vec;
}

vector_t flatten(const matrix3d_t& matrix)
{
  vector_t vec(matrix.size() * matrix[0].size() * matrix[0][0].size());
  for(std::size_t i = 0; i < matrix.size(); ++i)
  {
    for(std::size_t j = 0; j < matrix[i].size(); ++j)
    {
      std::copy(matrix[i][j].begin(), matrix[i][j].end(), vec.begin() + i * matrix[i].size() * matrix[i][j].size() + j * matrix[i][j].size());
    }
  }
  return vec;
}

matrix_t reshape(const vector_t& vec, std::size_t rows, std::size_t cols)
{
  matrix_t matrix(rows, vector_t(cols));
  for(std::size_t i = 0; i < rows; ++i)
  {
    std::copy(vec.begin() + i * cols, vec.begin() + (i + 1) * cols, matrix[i].begin());
  }
  return matrix;
}

matrix3d_t reshape(const vector_t& vec, std::size_t channels, std::size_t rows, std::size_t cols)
{
  matrix3d_t matrix(channels, matrix_t(rows, vector_t(cols)));
  for(std::size_t i = 0; i < channels; ++i)
  {
    for(std::size_t j = 0; j < rows; ++j)
    {
      std::copy(vec.begin() + i * rows * cols + j * cols, vec.begin() + i * rows * cols + (j + 1) * cols, matrix[i][j].begin());
    }
  }
  return matrix;
}

END_NAMESPACE