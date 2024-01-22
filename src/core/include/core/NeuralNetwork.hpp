#pragma once

#include <stdexcept>
#include <string>

#include "exceptions.hpp"
#include "Layer.hpp"
#include <data/data.hpp>


START_NAMESPACE_NEURAL_NETWORK

class NetStructure{

public:
    NetStructure() = default;
    ~NetStructure(){
        delete out;
        for (int i = 0; i < size; ++i){
            delete hidden[i];
        }
        delete [] hidden;
    }

    NetStructure(
        std::vector<int> structure,
        OutputLayer* out,
        Layer** hidden,
        int size
    ):
    structure(structure),
    out(out), size(size),
    hidden(hidden) {}

    std::vector<int> structure;
    OutputLayer* out;
    Layer** hidden;
    int size;

    bool operator==(const NetStructure& other);
    bool operator!=(const NetStructure& other);
};

class NeuralNetwork
{
    std::vector<int> _structure;

    double _curr_loss;
    double _average_loss;
    
    data::Data input;

    void raw_learn(const data::Data& data);
    void assert_data_size(const data::Data& data);

public:

    NeuralNetwork(const std::vector<int>& structure);

    /// @brief Copies data from NetStructure
    /// @param NetStructure* not nullptr
    NeuralNetwork(NetStructure*);

    ~NeuralNetwork();

    void
    set_input(data::Data& input);

    void
    raw_input(const std::vector<double>& inputs);

    /// @brief Runs learning process for a single data point, but it doesn't apply the learning, call `apply()` to commit 
    /// changes to the network
    /// @param data_point 
    void
    learn(data::Data& data_point);

    /// @brief Starts learning process for all of data in given input,
    /// applies the learning every time iterator reaches `apply_batch`
    /// @throw runtime_error if number of expected values are different from number of output neurons
    /// @param batch
    void
    learn(std::vector<data::Data>& batch, size_t apply_batch = 32);

    /// @brief Applies gradient to neurons
    /// @param learn_rate size of 'step'
    /// @param batch_size total size of data set
    void
    apply(double learn_rate = 1.2, int batch_size = 1);

    /// @brief Calculate cost of neural network for given data
    /// @return 
    double
    cost();

    /// @brief Calculates average loss on given data
    /// @param batch_size 
    /// @return 
    double
    loss(int batch_size);

    /// @brief Resets average loss to 0
    void 
    reset_loss();

    /// @brief Calculates the output values for the network
    /// @warning Should be called after setting inputs
    void
    output();

    /// @brief Returns copy of activation values of output neurons
    std::vector<double>
    get_outputs();

    /// @brief Returns index of the neurons with highest activation value
    /// @warning should be called after calling output()
    int
    classify();

    /// @brief Checks if it's guess was correct
    bool
    correct();

    /// @brief Copies this network's structure to NetStructure
    /// @warning allocated via new operator, should be deleted
    [[nodiscard]]
    NetStructure*
    structure();

    const std::vector<int>&
    raw_structure() {
        return _structure;
    }

    bool operator==(NeuralNetwork& other);
    bool operator==(const NetStructure& other);
    
    bool operator!=(NeuralNetwork& other);
    bool operator!=(const NetStructure& other);

    OutputLayer* _output_layer;
    Layer** _hidden_layer;
    int _hidden_size;
};

END_NAMESPACE