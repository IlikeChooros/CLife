#pragma once

#include <stdexcept>
#include <string>

#include "Layer.hpp"
#include "../data/data.hpp"

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
};

class NeuralNetwork
{
    OutputLayer* _output_layer;
    Layer** _hidden_layer;
    int _hidden_size;

    std::vector<int> _structure;

    double _curr_loss;
    double _average_loss;
    
    data::Data input;
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

    /// @brief Runs learning process for a single data point
    /// @param data_point 
    void
    learn(data::Data& data_point);

    /// @brief Starts learning process for all of data in given input
    /// @throw runtime_error if number of expected values are different from number of output neurons
    /// @param batch
    void
    learn(const std::vector<data::Data>& batch);

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
};