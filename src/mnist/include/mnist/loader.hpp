#pragma once

#include <fstream>
#include <vector>
#include <algorithm> 

#include <mnist/namespaces.hpp>

START_NAMESPACE_MNIST

constexpr int32_t MNIST_MAGIC_NUMBER = 2051;
constexpr int32_t MNIST_LABEL_MAGIC_NUMBER = 2049;

constexpr char MNIST_TRAINING_SET_IMAGE_FILE_NAME[] = "train-images-idx3-ubyte";
constexpr char MNIST_TRAINING_SET_LABEL_FILE_NAME[] = "train-labels-idx1-ubyte";
constexpr char MNIST_TEST_SET_IMAGE_FILE_NAME[] = "t10k-images-idx3-ubyte";
constexpr char MNIST_TEST_SET_LABEL_FILE_NAME[] = "t10k-labels-idx1-ubyte";

class Loader{

    std::string path;

public:
    Loader(const std::string& path);

    Loader& load(const std::string& path);

    /**
     * @brief Get the images object, normalized to [0, 1]
    */
    std::vector<std::vector<float>> get_images();
    std::vector<std::vector<float>> get_labels();
};

END_NAMESPACE_MNIST