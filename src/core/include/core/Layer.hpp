#pragma once

#include "Neuron.hpp"
#include <vector>

START_NAMESPACE_NEURAL_NETWORK

class BaseLayer
{

public:
    BaseLayer() = default;
    BaseLayer(BaseLayer*);
    ~BaseLayer();
    
    /// @brief Calculates neuron activation values
    /// @warning Inputs should be set before calling this method
    /// @return outputs
    const std::vector<double>&
    calc_outputs();

    void
    apply_gradient(double learn_rate, int batch_size);

    void
    set_inputs(const std::vector<double>& inputs);

    std::vector<double> outputs;
    std::vector<double> inputs;

    BaseNeuron** neurons;

    std::vector<double> output_val;
    int node_in;
    int node_out;
};


class Layer: public BaseLayer
{
    public:
    Layer(int inputs, int outputs, BaseActivation* act = nullptr);
    Layer() = default;
    Layer(Layer* other);

    void
    create(int inputs, int outputs, BaseActivation* act = nullptr);

    Layer*
    calc_gradient
    (BaseLayer* prev_layer, const std::vector<double>& node_consts);

};

class OutputLayer: public BaseLayer
{
    public:
    OutputLayer(int inputs, int outputs, BaseActivation* act = nullptr);
    OutputLayer() = default;
    OutputLayer(OutputLayer* other);

    void
    create(int inputs, int outputs, BaseActivation* act = nullptr);


    BaseLayer*
    calc_gradient(const std::vector<double>& expected);


    /// @brief Calculates the cost of the layer
    /// @param expected 
    /// @return sum of output neuron errors (using function: (activation - expected_value)^2)
    double
    cost(const std::vector<double>& expected);

};

END_NAMESPACE