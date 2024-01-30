#pragma once

#include "loader.hpp"
#include <random>

START_NAMESPACE_MNIST

class transformator{
    public:
    transformator() = default;

    /**
     * @brief Transform the data into a vector of Data objects
    */
    data::data_batch* transform(
        std::vector<std::vector<double>>* images,
        std::vector<std::vector<double>>* labels
    );

    /**
     * @brief Add noise to the data, the noise is a random vector, applied for each pixel.
     * Also changes some pixel values at random.
    */
    data::data_batch* add_noise(
        data::data_batch* data, 
        std::uniform_real_distribution<double>& dist,
        std::mt19937& gen
    );
};

END_NAMESPACE_MNIST