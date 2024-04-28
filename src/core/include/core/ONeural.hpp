#pragma once

#include <algorithm>
#include <iostream>

#include <data/data.hpp>
#include "OLayer.hpp"
#include "thread.hpp"

/*

optimized neural network, using only layers

*/

START_NAMESPACE_NEURAL_NETWORK

using data_batch = data::data_batch;

struct _NetworkFeedData{
    _NetworkFeedData() = default;
    _NetworkFeedData(OLayer& output, std::vector<OLayer>& hidden) {
        _layer_feed_data.reserve(hidden.size() + 1);
        for (auto& layer : hidden){
            _layer_feed_data.emplace_back(layer._inputs_size, layer._neurons_size);
        }
        _layer_feed_data.emplace_back(output._inputs_size, output._neurons_size);
    }
    _NetworkFeedData& setInputs(vector_t& inputs){
        _layer_feed_data[0]._inputs = inputs;
        return *this;
    }
    std::vector<_FeedData> _layer_feed_data;
};

/// @brief optimized neural network
class ONeural{

    // Mutex used for locking `_loss` and `_cost` when updating them
    std::mutex _mutex;

    /*
    Feeds forward the network with `data` and calculates `gradient_weights` and
    `gradient_baises` for the layers. The network may be updated with `apply(...)` call
    
    @param data single data point
    @param context Neural network pointer
    */
    static void _update_gradients(data::Data&& data, ONeural* context);

    /*
    Creates for every `Data` instance in the `tranining_data` new thread,
    and calls `_update_gradients(...)`

    @param training_data mini-batch data, shouldn't be too big (go for 16)
    @param learn_rate learning rate, make it small, since the batch size is also small
    */
    void _learn_multithread(data_batch* training_data, double learn_rate);

    size_t _accuracy_multithread(data_batch* mini_test);
    size_t _classify_feed(_NetworkFeedData&);
    bool _correct_feed(_NetworkFeedData&, vector_t& expect);

    size_t _iterator;

    public:
    ONeural() = default;

    /// @brief calls internally `build(...)`
    /// @throw invalid_structure if the user gives invalid structure (.size() < 2)
    /// @param structure structures of the neural network, ex. {2,5,2,4} ->
    /// 2 inputs, 4 outputs, (5,2) -> number of neurons in hidden layers (2 hidden layers)
    ONeural(
        const std::vector<size_t>& structure, 
        ActivationType output_activation = ActivationType::softmax,
        ActivationType hidden_activation = ActivationType::relu,
        double dropout_rate = 0.0
    );

    /// @brief builds the neural network with given structure
    /// @param structure structures of the neural network, ex. {2,5,2,4} ->
    /// 2 inputs, 4 outputs, (5,2) -> number of neurons in hidden layers (2 hidden layers)
    /// @return *this
    ONeural& build(
        const std::vector<size_t>& structure, 
        ActivationType output_activation = ActivationType::softmax,
        ActivationType hidden_activation = ActivationType::relu,
        double dropout_rate = 0.0
    );

    /// @brief Initializes the weights and biases with random values,
    /// may be called only after `build()`
    void initialize();

    /// @brief Sets the network to training mode, dropout is applied
    void training_mode(bool mode = true);

    /// @brief Trains the network on given `data`, doesn't apply gradients
    /// @param data single data point
    void train(data::Data& data);

    /// @brief Learns signle point -> calculates gradients and applies them
    /// @param data single data point
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
    void raw_input(const vector_t& _raw_input);

    /// @brief feed forward the network, calculate activations on the layers
    void feed_forward(_NetworkFeedData& feed_data, vector_t& inputs);

    /// @brief Calculates the outputs of the network
    /// @return activations of the output layer
    vector_t outputs();

    /// @brief average loss of the network on given `batch_size`
    /// @param batch_size 
    real_number_t loss(size_t batch_size = 32UL);

    /// @brief current cost of the network, outputs must be already calculated
    real_number_t cost();

    /// @brief Tell wheter the network's guess was correct, outputs must be already calculated
    bool correct();

    /// @brief return the index of the most activated neuron, outputs must be already calculated
    size_t classify();

    /// @brief return the structure of the network
    const std::vector<size_t>& structure();

    /// @brief Set the activation functions for the network
    /// @param output_activation 
    /// @param hidden_activation 
    void activations(
        ActivationType output_activation,
        ActivationType hidden_activation
    );

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

    real_number_t accuracy(data_batch* test);

    OLayer _output_layer;
    std::vector<OLayer> _hidden_layers;

    real_number_t _cost;
    real_number_t _loss;
    data::Data _input;
    vector_t _outputs;
    std::vector<size_t> _structure;
};


END_NAMESPACE