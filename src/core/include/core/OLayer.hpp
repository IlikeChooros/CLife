/*

Heavily optimized Neural Network

Using only layer (without Neurons as an individual node)

*/
#pragma once

#include <memory>
#include <random>
#include <cmath>
#include <chrono>

#include <data/data.hpp>
#include "namespaces.hpp"
#include "activation.hpp"
#include "exceptions.hpp"
#include "types.hpp"

START_NAMESPACE_NEURAL_NETWORK



class OLayer{
    
    friend class ONeural;

    void _match_activations(const ActivationType& activation);
    inline double _derivative(size_t index);

    public:
    OLayer() = default;

    /// @brief Calls `build(...)` internally
    OLayer(size_t inputs, size_t outputs, ActivationType&& type = ActivationType::sigmoid);

    /// @brief Builds the structure, reserves memory for this layer: allocates weights and biases
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
    vector_t& calc_activations(vector_t&& inputs);
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
    OLayer* calc_output_gradient(vector_t&& expected);

    /// @warning first call `calc_hidden_gradient` or `calc_output_gradient`
    /// @brief Updates the graidents: weight, bias values. Call this before applying them
    void update_gradients();

    /// @brief This does excacly what you think it does.
    /// @param learn_rate 
    /// @param batch_size 
    void apply_gradients(double learn_rate, size_t batch_size);

    /**
     * @brief Calculates the cost of the output layer given the expected output.
     * @param expected A vector of expected output values.
     * @return The calculated cost.
     */
    real_number_t cost(vector_t&& expected);

    /**
     * @brief Returns the activations of the neurons in the layer.
     * @return A reference to the vector of neuron activations.
     */
    vector_t& activations();

    /**
     * @brief Returns the weight of a connection from a specific input to a specific neuron.
     * @param inputIdx The index of the input.
     * @param neuronIdx The index of the neuron.
     * @return A constant reference to the weight.
     */
    const real_number_t& weight(size_t inputIdx, size_t neuronIdx);
    
    /**
     * @brief Overloads the assignment operator for the OLayer class.
     * @param other The OLayer object to be copied.
     * @return A reference to the current object.
     */
    OLayer& operator=(const OLayer& other);
    bool operator==(const OLayer& other);
    bool operator!=(const OLayer& other);

    size_t _neurons_size;
    size_t _inputs_size;
    matrix_t _weights;
    matrix_t _gradient_weights;

    vector_t _biases;
    vector_t _gradient_biases;
    vector_t _activations;
    vector_t _inputs;
    vector_t _weighted_inputs;
    vector_t _partial_derivatives;
    // momentum gradient
    matrix_t _m_gradient;
    vector_t _m_gradient_bias;
    // velocity gradient
    matrix_t _v_gradient;
    vector_t _v_gradient_bias;
    
    std::function<real_number_t(vector_t&, size_t i)> _activation_function;
    std::function<real_number_t(vector_t&, size_t i)> _derivative_of_activ;
    ActivationType _activ_type;
};


END_NAMESPACE
