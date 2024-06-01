#pragma once

#include "loader.hpp"
#include <random>

START_NAMESPACE_MNIST

/// @brief Class for adding noise to the data
class transformator
{
public:
    transformator() = default;

    /**
     * @brief
     * Add noise to the data:
     *  * Moves the pixels at random.
     *  * Rotates the pixels at random.
     *  * Adds random noise to the pixels.
     *
     * @param data The data to add noise to
     * @param max_vector maximum noise vector lenght (shouldn't be bigger than 1/4 of the image width/height)
     * @return new `noisy` data
     */
    data::data_batch *add_noise(
        data::data_batch *data,
        int max_vector = 4,
        size_t cols = 28,
        size_t rows = 28,
        size_t noisiness = 100);

    /**
     * @brief Rotate the pixels by angle and given center point
     * @returns new vector of pixels
     */
    data::vector_t rotate(
        data::vector_t &pixels,
        size_t cols = 28,
        size_t rows = 28,
        float angle = 30.0f,
        int center_x = 14,
        int center_y = 14);

    /**
     * @brief Move the pixels by x and y
     * @returns new vector of pixels
     */
    data::vector_t move(
        data::vector_t &pixels,
        size_t cols = 28,
        size_t rows = 28,
        int x = 0,
        int y = 0);
};

END_NAMESPACE_MNIST