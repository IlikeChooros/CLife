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

    void _update_gradients(data::Data&& data);
    size_t _iterator;

    public:
    ONeural() = default;

    /// @brief calls internally `build(...)`
    /// @throw invalid_structure if the user gives invalid structure (.size() < 2)
    /// @param structure structures of the neural network, ex. {2,5,2,4} ->
    /// 2 inputs, 4 outputs, (5,2) -> number of neurons in hidden layers (2 hidden layers)
    ONeural(
        const std::vector<double>& structure, 
        ActivationType output_activation = ActivationType::sigmoid,
        ActivationType hidden_activation = ActivationType::relu
    );

    /// @brief builds the neural network with given structure
    /// @param structure structures of the neural network, ex. {2,5,2,4} ->
    /// 2 inputs, 4 outputs, (5,2) -> number of neurons in hidden layers (2 hidden layers)
    /// @return *this
    ONeural& build(
        const std::vector<double>& structure, 
        ActivationType output_activation = ActivationType::sigmoid,
        ActivationType hidden_activation = ActivationType::relu
    );

    /// @brief Initializes the weights and biases with random values,
    /// may be called only after `build()`
    /// @return *this
    ONeural& initialize();

    /// @brief Trains the network on given `data`, doesn't apply gradients
    /// @param data single data point
    void train(data::Data& data);

    /// @brief Learns signle point -> calculates gradients and applies them
    /// @param data signle data point
    /// @param learn_rate learning rate
    void learn(data::Data& data, double learn_rate = 0.4);

    /// @brief Learns whole given `training_data` and applies the average graident 
    /// calculated on the whole batch
    /// @param training_data training batch
    /// @param learn_rate learning rate
    void learn(data_batch* training_data, double learn_rate = 0.4);

    /// @brief Learns by mini batches given `whole_data`, divides the `whole_data` into smaller chunks
    /// with the size of `batch_size`, and calls `learn(data_batch* training_data, double learn_rate)`
    /// internally. Call this method repeatedly to let the network iterate over whole data (whole_data.size() / batch_size times)
    /// @param whole_data whole training data
    /// @param learn_rate learning rate
    /// @param batch_size mini batch size
    void batch_learn(data_batch* whole_data, double learn_rate = 0.4, size_t batch_size = 32UL);

    /// @brief Applies the gradients calculated by `train(...)` method
    /// @param learn_rate learning rate 
    /// @param batch_size batch size of the training data
    void apply(double learn_rate = 0.4, size_t batch_size = 32UL);

    /// @brief set input for then network, inputs must be normalized, data.input values should range in <-1, 1>
    /// @param data 
    void input(data::Data& data);

    /// @brief set raw input, you can not call any of the learning, or cost evaulation
    /// methods, since expected output of the network was not set for this input
    /// @param _raw_input 
    void raw_input(const std::vector<double>& _raw_input);

    /// @brief Calculates the outputs of the network
    /// @return activations of the output layer
    const std::vector<double>& outputs();

    /// @brief average loss of the network on given `batch_size`
    /// @param batch_size 
    double loss(size_t batch_size = 32UL);

    /// @brief current cost of the network, outputs must be already calculated
    double cost();

    /// @brief Tell wheter the network's guess was correct, outputs must be already calculated
    bool correct() const;

    /// @brief return the index of the most activated neuron, outputs must be already calculated
    size_t classify() const;

    /// @brief return the structure of the network
    const std::vector<double>& structure();

    /// @brief deep compare neural network with `other`
    /// @return true if the networks are equal, otherwise false
    bool operator==(const ONeural& other);

    /// @brief deep compare neural network with `other`
    /// @return false if the networks are equal, otherwise true
    bool operator!=(ONeural& other);

    /// @brief Copies from other to this.
    /// @param other 
    /// @return *this
    ONeural& operator=(const ONeural& other);

    double accuracy(data_batch* test);

    OLayer _output_layer;
    std::vector<OLayer> _hidden_layers;

    double _cost;
    double _loss;
    data::Data _input;
    std::vector<double> _outputs;
    std::vector<double> _structure;
};


END_NAMESPACE