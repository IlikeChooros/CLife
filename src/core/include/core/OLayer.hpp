/*

Heavily optimized Neural Network

Using only layer (without Neurons as an individual node)

*/
#pragma once

#include "namespaces.hpp"
#include "Activation/activation.hpp"

#include <vector>
#include <memory>
#include <random>

START_NAMESPACE_NEURAL_NETWORK

class OLayer{
    
    friend class ONeural;

    std::vector<std::vector<double>> _weights;
    std::vector<std::vector<double>> _gradient_weights;

    std::vector<double> _biases;
    std::vector<double> _gradient_biases;
    std::vector<double> _activations;
    std::vector<double> _inputs;
    std::vector<double> _partial_derivatives;

    std::function<double(double)> _activation_function;
    std::function<double(double)> _derivative_of_activ;

    public:
    OLayer() = default;

    /// @brief Calls `build(...)` internally
    OLayer(size_t inputs, size_t outputs, ActivationType&& type = ActivationType::sigmoid);

    /// @brief Builds the structure, sets reserves memory for this layer: allocates weights and biases
    /// @param inputs number of inputs for the layer
    /// @param outputs number of outputs 
    /// @param type type of activation function
    /// @return *this
    OLayer& build(size_t inputs, size_t outputs, ActivationType&& type = ActivationType::sigmoid);

    /// @brief Initializes weights and biases with random values
    /// @return *this
    OLayer& initialize();

    /// @brief This does excacly what you think it does. Call this before calculating gradients
    /// @param inputs 
    /// @return activation values
    std::vector<double>& calc_activations(std::vector<double>&& inputs);
    void calc_activations();


    /// @brief Calculates hidden layer gradient values, based on backpropagation algorithm
    /// @param prev_layer previously evaulated layer
    /// @warning first call `calc_activations`
    /// @return this pointer
    OLayer* calc_hidden_gradient(OLayer* prev_layer);

    /// @brief Calculates output layer gradient values
    /// @param expected expected activation values
    /// @warning first call `calc_activations`
    /// @return this pointer
    OLayer* calc_output_gradient(std::vector<double>&& expected);

    /// @warning first call `calc_hidden_gradient` or `calc_output_gradient`
    /// @brief Updates the graidents: weight, bias values. Call this before applying them
    void update_gradients();

    /// @brief This does excacly what you think it does.
    /// @param learn_rate 
    /// @param batch_size 
    void apply_gradients(double learn_rate, size_t batch_size);

    double cost(std::vector<double>&& expected);
    std::vector<double>& activations();
    const double& weight(size_t inputIdx, size_t neuronIdx);
    
    OLayer& operator=(const OLayer& copy);
};


END_NAMESPACE
