#pragma once

#include <algorithm>

#include <data/data.hpp>

#include "OLayer.hpp"
#include "exceptions.hpp"

/*

optimized neural network, using only layers

*/

START_NAMESPACE_NEURAL_NETWORK

using data_batch = std::vector<data::Data>;

/// @brief optimized neural network
class ONeural{
    OLayer _output_layer;
    std::vector<OLayer> _hidden_layers;

    double _cost;
    double _loss;
    data::Data _input;
    std::vector<double> _outputs;

    void _update_gradients(data::Data&& data);

    public:
    ONeural() = default;
    /// @brief calls internally `build(...)`
    /// @throw invalid_structure if the user gives invalid structure (.size() < 2)
    /// @param structure structures of the neural network, ex. {2,5,2,4} ->
    /// 2 inputs, 4 outputs, (5,2) -> number of neurons in hidden layers (2 hidden layers)
    ONeural(const std::vector<double>& structure);

    /// @brief builds the neural network with given structure
    /// @param structure structures of the neural network, ex. {2,5,2,4} ->
    /// 2 inputs, 4 outputs, (5,2) -> number of neurons in hidden layers (2 hidden layers)
    /// @return *this
    ONeural& build(const std::vector<double>& structure);

    /// @brief Initializes the weights and biases with random values,
    /// may be called only after `build()`
    /// @return *this
    ONeural& initialize();

    
    void learn(data::Data& data, double learn_rate = 0.4);
    void learn(data_batch* training_data, double learn_rate = 0.4);
    void input(data::Data& data);
    void raw_input(const std::vector<double>& _raw_input);
    const std::vector<double>& outputs();
    double loss(size_t batch_size);
    double cost();
    bool correct();
    size_t classify();
};


END_NAMESPACE