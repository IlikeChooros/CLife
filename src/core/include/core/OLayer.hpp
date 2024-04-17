/*

Heavily optimized Neural Network

Using only layer (without Neurons as an individual node)

*/
#pragma once

#include <memory>
#include <random>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>

#include <data/data.hpp>
#include "namespaces.hpp"
#include "activation.hpp"
#include "exceptions.hpp"
#include "utils.hpp"

START_NAMESPACE_NEURAL_NETWORK

/// @brief Data structure for backpropagation algorithm, holds
/// all necessary data for calculating gradients, used for multithreading
struct _FeedData{
    public:
    _FeedData() = default;
    _FeedData(size_t inputs, size_t outputs);
    _FeedData& build(size_t inputs, size_t outputs);

    vector_t _activations;
    vector_t _inputs;
    vector_t _weighted_inputs;
    vector_t _partial_derivatives; 
};

/*

Heavily optimized Neural Network Layer

Layer is a collection of neurons, each neuron has multiple weights 
(fully connected to the previous) and a bias. The activation function
is applied to the sum of the weighted inputs and the bias.

In simple terms, layer is an array of neurons, each neuron has a set of weights
and a bias. 

(many connections - weights)
        ----\
        ----- (neuron, +bias) ---- (output)
        ----/

This implementation is heavily optimized for speed and memory usage.
Data is stored in a flatened matrix, has multithreading support.


*/
class OLayer{
    
    friend class ONeural;

    void _match_activations(const ActivationType& activation);
    inline vector_t _derivative(_FeedData& feed_data);

    // Mutex used for multithreading when accessing the `_gradient_weights` and `_gradient_biases`
    std::mutex _mutex;

    public:
    OLayer() = default;
    OLayer(const OLayer& other);

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
    vector_t& calc_activations(_FeedData& feed_data);
    // void calc_activations();


    /// @brief Calculates hidden layer gradient values, based on backpropagation algorithm
    /// @param prev_layer previously evaulated layer
    /// @warning first call `calc_activations`
    /// @return this pointer
    OLayer* calc_hidden_gradient(OLayer* prev_layer, _FeedData& feed_data, vector_t& _prev_partial_derivatives);

    /// @brief Calculates output layer gradient values
    /// @param expected expected activation values
    /// @warning first call `calc_activations`
    /// @return this pointer
    OLayer* calc_output_gradient(vector_t&& expected, _FeedData& feed_data);

    /// @warning first call `calc_hidden_gradient` or `calc_output_gradient`
    /// @brief Updates the graidents: weight, bias values. Call this before applying them
    void update_gradients(_FeedData& feed_data);

    /// @brief This does excacly what you think it does.
    /// @param learn_rate 
    /// @param batch_size 
    void apply_gradients(double learn_rate, size_t batch_size);

    /**
     * @brief Calculates the cost of the output layer given the expected output.
     * @param expected A vector of expected output values.
     * @return The calculated cost.
     */
    real_number_t cost(vector_t&& expected, _FeedData& feed_data);


    /**
     * @brief Returns the weight of a connection from a specific input to a specific neuron.
     * @param inputIdx The index of the input.
     * @param neuronIdx The index of the neuron.
     * @return A constant reference to the weight.
     */
    inline const real_number_t& weight(size_t inputIdx, size_t neuronIdx){
        return _weights[neuronIdx * _inputs_size + inputIdx];
    }
    
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
    vector_t _weights; // flatened matrix
    vector_t _gradient_weights; // flatened matrix

    vector_t _biases;
    vector_t _gradient_biases;
    
    // momentum gradient
    vector_t _m_gradient; // flatened matrix
    vector_t _m_gradient_bias;

    // velocity gradient
    vector_t _v_gradient; // flatened matrix
    vector_t _v_gradient_bias;
    
    std::function<vector_t(vector_t&)> _activation_function;
    std::function<vector_t(vector_t&)> _derivative_of_activ;
    ActivationType _activ_type;
};


END_NAMESPACE
