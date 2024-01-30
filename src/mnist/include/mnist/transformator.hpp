#pragma once

#include "loader.hpp"
#include <random>

START_NAMESPACE_MNIST

class transformator{
    public:
    transformator() = default;

    /**
     * @brief Add noise to the data, the noise is a random vector, applied for each pixel.
     * Also changes some pixel values at random.
     * 
     * @param data The data to add noise to
     * @param max_vector maximum noise vector lenght (shouldn't be bigger than 1/4 of the image width/height)
     * @return new `noisy` data
    */
    data::data_batch* add_noise(
        data::data_batch* data, 
        int max_vector = 6,
        size_t cols = 28,
        size_t rows = 28,
        size_t noisiness = 80
    );
};

END_NAMESPACE_MNIST