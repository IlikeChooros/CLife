#include <core/OLayer.hpp>

START_NAMESPACE_NEURAL_NETWORK

OLayer::OLayer(size_t inputs, size_t outputs, ActivationType&& type){
    build(inputs, outputs, std::forward<ActivationType>(type));
}

OLayer& OLayer::build(size_t inputs, size_t outputs, ActivationType&& type){

    _activation_function = matchActivation(std::forward<ActivationType>(type));
    _derivative_of_activ = matchDerivative(std::forward<ActivationType>(type));

    _weights.assign(outputs, std::vector<double>());
    _gradient_weights.assign(outputs, std::vector<double>());

    for (size_t i = 0; i < outputs; i++){
        _weights[i].assign(inputs, 0);
        _gradient_weights[i].assign(inputs, 0);
    }
    _biases.assign(outputs, 0);
    _gradient_biases.assign(outputs, 0);
    _activations.assign(outputs, 0);
    _partial_derivatives.assign(outputs, 0);

    _neurons_size = outputs;
    _inputs_size = inputs;

    return *this;
}

OLayer& OLayer::initialize(){
    std::default_random_engine _engine;

    std::uniform_real_distribution<double> _dist(-1, 1);

    for (size_t i = 0; i < _weights.size(); i++){
        for(size_t j = 0; j < _weights[i].size(); j++){
            _weights[i][j] = _dist(_engine);
        }
        _biases[i] = _dist(_engine);
    }

    return *this;
}

std::vector<double>& OLayer::calc_activations(std::vector<double>&& inputs){
    _inputs = inputs;
    calc_activations();
    return _activations;
}

void OLayer::calc_activations(){
    // assuming that inputs are already set

    // :))))
    double neuron_activation;
    for (size_t i = 0; i < _neurons_size; i++){

        // using equasion:
        // weighted_input = input * weight + bias
        // For mulitple, it is just a sum of all weighted_inputs

        neuron_activation = _biases[i];
        for (size_t j = 0; j < _inputs_size; j++){
            neuron_activation += _weights[i][j] * _inputs[j];
        }
        _activations[i] = _activation_function(neuron_activation);
    }
}

OLayer* OLayer::calc_hidden_gradient(OLayer* prev_layer){

    /*
    L - previous layer
    L - 1 -> current layer

    d(cost)/d(weight_(L-1)) = d(cost)/d(activation_(L-1)) * d(activation_(L-1))/d(weighted_input) * d(weighted_input_(L-1))/d(weight_(L-1)) 

    d(activation_(L-1))/d(weighted_input) = _derivative_of_activ

    d(weighted_input_(L-1))/d(weight_(L-1)) = activation_(L-2) (inputs)
    
    d(cost)/d(activation_(L-1)) = d(cost)/d(activation_L) * d(activation_L)/d(weighted_input_L) * d(weighted_input_L)/d(activation_(L-1))
                                    <--------------------------------------------------------->
                                                    _partial_derivatives ( L )
                                = _partial_derivatives_L * _weight_L
    
    so final answer is:
        d(cost)/d(weight_(L-1)) = _partial_derivatives_L * _weight_L * _derivative_of_activ * activation_(L-2)

    similarily:
        d(cost)/d(bias_(L-1)) = _partial_derivatives_L * _weight_L * _derivative_of_activ * 1


    You can see that 
        new_partial_derviative = _partial_derivatives_L * _weight_L * _derivative_of_activ
    
    That is common factor for both of the derivatives, so it useful to store this data.
    In update_gradients:
        when calculating gardient weight im multiplying _partial_derivatives by input,
        essentialy getting the answer: 
            _gradient_weights = _partial_derivatives * input 


    */

    size_t neurons = _weights.size();

    for (size_t n = 0; n < _neurons_size; n++){
        double new_partial_derviative = 0;
        for (size_t prev = 0; prev < prev_layer->_neurons_size; prev++){
            new_partial_derviative += prev_layer->weight(n, prev) * prev_layer->_partial_derivatives[prev];
        }
        new_partial_derviative *= _derivative_of_activ(_activations[n]);
        _partial_derivatives[n] = new_partial_derviative;
    }

    return this;
}


OLayer* OLayer::calc_output_gradient(std::vector<double>&& expected){

    double error_deriv, deriv_with_error_and_activation;

    for (size_t n = 0; n < _neurons_size; n++){
        error_deriv = 2 * (_activations[n] - expected[n]);

        // Calculate partial derivative: d(cost)/d(activation) * d(activation)/d(weighted_input)
        deriv_with_error_and_activation = error_deriv * _derivative_of_activ(_activations[n]);
        _partial_derivatives[n] = deriv_with_error_and_activation;
    }
    return this;
}

void OLayer::update_gradients(){
    for (size_t i = 0; i < _neurons_size; i++){

        // partial derivatives are calculated in `calc_output_gradient` and `calc_hidden_gradient`
        
        for(size_t j = 0; j < _inputs_size; j++){
            /* 
            That is complete derviative:
                For output layer:
                    d(cost)/d(weight) = d(cost)/d(activation) * d(activation)/d(weighted_input) * d(weighted_input)/d(weight)
                                        <--------------------------------------------------->
                                                        partial derivative
                                      = 2 * (activation - expected) * (derivative of the activation function) * input

                For hidden layers see `calc_hidden_gradient`
            */
            _gradient_weights[i][j] += _partial_derivatives[i] * _inputs[j];
        }

        /* 
        Similarily:
            For ouptut layer:
                d(cost)/d(bias) = d(cost)/d(activation) * d(activation)/d(weighted_input) * d(weighted_input)/d(bias)
                                = 2 * (activation - expected) * (derivative of the activation function) * 1
        */
        _gradient_biases[i] += _partial_derivatives[i];
    }

}

void OLayer::apply_gradients(double learn_rate, size_t batch_size){

    double weighted_learn_rate = learn_rate / double(batch_size);

    for (size_t n = 0; n < _neurons_size; n++){
        for(size_t w = 0; w < _inputs_size; w++){
            _weights[n][w] -= _gradient_weights[n][w] * weighted_learn_rate;
            _gradient_weights[n][w] = 0;
        }
        _biases[n] -= _gradient_biases[n] * weighted_learn_rate;
        _gradient_biases[n] = 0;
    }
}

double OLayer::cost(std::vector<double>&& expected){
    double error, cost = 0;

    for (size_t i = 0; i < _neurons_size; i++){
        error = _activations[i] - expected[i];
        cost += error*error;
    }
    return cost;
}

std::vector<double>& OLayer::activations(){
    return _activations;
}

const double& OLayer::weight(size_t inputIdx, size_t neuronIdx){
    return _weights[neuronIdx][inputIdx];
}

OLayer& OLayer::operator=(const OLayer& other){
    _weights = other._weights;
    _gradient_weights = other._gradient_weights;
    _biases = other._biases;
    _gradient_biases = other._gradient_biases;
    _activations = other._activations;
    _inputs = other._inputs;
    _partial_derivatives = other._partial_derivatives;

    _activation_function = other._activation_function;
    _derivative_of_activ = other._derivative_of_activ;
    return *this;
}

// returns true if the a != b, else (false) a == b.
inline bool diffrent_double(double a, double b, double approx = 0.001){
    return a - b > approx;
}

bool OLayer::operator==(const OLayer& other){
    if (other._neurons_size != _neurons_size){
        return false;
    }
    
    if(other._inputs_size != _inputs_size){
        return false;
    }

    for (size_t i = 0; i < _neurons_size; i++){
        if (diffrent_double(_biases[i], other._biases[i])){
            return false;
        }
        for (size_t j = 0;j < _inputs_size; j++){
            if (diffrent_double(_weights[i][j], other._weights[i][j])){
                return false;
            }
        }
    }
    return true;
}

bool OLayer::operator!=(const OLayer& other){
    return !this->operator==(other);
}

END_NAMESPACE