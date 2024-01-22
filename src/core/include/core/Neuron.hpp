#pragma once

#include "namespaces.hpp"
#include "../../../random-gen/Random.hpp"
#include "./Activation/activation.hpp"

#include <vector>
#include <memory>

START_NAMESPACE_NEURAL_NETWORK

class BaseNeuron
{

protected:
    std::vector<double> gradient_weights;
    double gradient_bias;

    std::vector<double> _inputs;

    int _conn;
    double _activation;

    void
    construct(int conn);

    Random gen;

    BaseActivation* _activationFunctor;

public:
    explicit BaseNeuron(int connections, BaseActivation* act = nullptr);

    BaseNeuron() = default;


    BaseNeuron(BaseNeuron*);

    virtual ~BaseNeuron();

    /// @brief Calculates activation value, using sigmoid function
    double 
    activation();

    virtual double
    calculate_gradient(const double& deriv_) = 0;

    void 
    apply_gradients(const double& learnrate = 1.1, const int& batch_size = 1);

    /// @brief Calculates sqared difference between activation value and expected one 
    /// @return 
    double
    error(const double& expected);

    void 
    set_inputs(const std::vector<double>& inputs);
    void 
    set_inputs(double* inputs);

    void
    create(int conn);

    /// @brief Returns number of inputs for this neuron
    /// @return  
    int
    connections();

    void printNeuron();

    std::vector<double> weights;
    double bias;
};


class Neuron: public BaseNeuron
{
    public:
    Neuron(int connections, BaseActivation* act = nullptr):
    BaseNeuron(connections, act) {}

    Neuron() = default;

    Neuron(Neuron* other): 
    BaseNeuron(other) {}

    Neuron(BaseNeuron* other):
    BaseNeuron(other) {}

    /// @brief Calculates gradient for this neuron
    /// @param deriv_ 
    /// @return 
    double
    calculate_gradient(const double& deriv_);
};

class 
OutputNeuron: public BaseNeuron
{
public:

    OutputNeuron(int connections, BaseActivation* act = nullptr): 
    BaseNeuron(connections, act) {}

    OutputNeuron() = default;

    OutputNeuron(OutputNeuron* other): 
    BaseNeuron(other) {}

    OutputNeuron(BaseNeuron* other):
    BaseNeuron(other) {}
    
    /// @brief Calculates output gradient
    /// @return 
    double
    calculate_gradient(const double& expected);
};

END_NAMESPACE