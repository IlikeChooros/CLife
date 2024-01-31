#include <core/OLayer.hpp>

START_NAMESPACE_NEURAL_NETWORK

OLayer::OLayer(size_t inputs, size_t outputs, ActivationType&& type){
    build(inputs, outputs, std::forward<ActivationType>(type));
}

OLayer& OLayer::build(size_t inputs, size_t outputs, ActivationType&& type){

    _match_activations(type);

    _weights.assign(outputs, std::vector<double>(inputs, 0));
    _gradient_weights.assign(outputs, std::vector<double>(inputs, 0));
    _m_gradient.assign(outputs, std::vector<double>(inputs, 0));
    _v_gradient.assign(outputs, std::vector<double>(inputs, 0));

    _biases.assign(outputs, 0);
    _gradient_biases.assign(outputs, 0);
    _activations.assign(outputs, 0);
    _partial_derivatives.assign(outputs, 0);
    _weighted_inputs.assign(outputs, 0);
    

    _neurons_size = outputs;
    _inputs_size = inputs;

    return *this;
}

OLayer& OLayer::initialize(){
    std::default_random_engine _engine(std::chrono::system_clock::now().time_since_epoch().count());
    double limit = sqrt(2.0 / _inputs_size);
    std::uniform_real_distribution<double> _dist(-limit, limit);

    for(auto& weights : _weights){
        // std::generate calls _dist(_engine) for each element in weights
        std::generate(weights.begin(), weights.end(), [&](){return _dist(_engine);});
    }
    std::generate(_biases.begin(), _biases.end(), [&](){return _dist(_engine);});

    return *this;
}

void OLayer::_match_activations(const ActivationType& activation){
    _activ_type = activation;
    switch (_activ_type)
    {
    case ActivationType::sigmoid:
        _activation_function = sigmoid::activation;
        _derivative_of_activ = sigmoid::derivative;
        break;
    case ActivationType::relu:
        _activation_function = relu::activation;
        _derivative_of_activ = relu::derivative;
        break;
    case ActivationType::softmax:
        _activation_function = softmax::activation;
        _derivative_of_activ = softmax::derivative;
        break;
    case ActivationType::silu:
        _activation_function = silu::activation;
        _derivative_of_activ = silu::derivative;
        break;
    case ActivationType::selu:
        _activation_function = selu::activation;
        _derivative_of_activ = selu::derivative;
        break;
    case ActivationType::prelu:
        _activation_function = prelu::activation;
        _derivative_of_activ = prelu::derivative;
        break;
    default:
        break;
    }
}

inline double OLayer::_derivative(size_t index){
    switch (_activ_type)
    {
    case ActivationType::silu:
    case ActivationType::selu:
        return _derivative_of_activ(_weighted_inputs, index);
    default:
        return _derivative_of_activ(_activations, index);
    }
}

std::vector<double>& OLayer::calc_activations(std::vector<double>&& inputs){
    _inputs = inputs;
    calc_activations();
    return _activations;
}


void OLayer::calc_activations(){
    // assuming that inputs are already set
    for (size_t i = 0; i < _neurons_size; i++){

        // using equasion:
        // weighted_input = input * weight + bias
        // For mulitple, it is just a sum of all weighted_inputs
        
        // std::inner_product calculates sum of all weighted_inputs, with starting value of bias
        // on some hardware it is faster than for loop
        _weighted_inputs[i] = std::inner_product(_weights[i].begin(), _weights[i].end(), _inputs.begin(), _biases[i]);
        
        
        _activations[i] = _activation_function(_weighted_inputs, i);
        //         if (std::isnan(_activations[i])){
        //     _activation_function(_weighted_inputs, i);
        //     while(1){}
        // }
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

    for (size_t n = 0; n < _neurons_size; n++){
        double new_partial_derviative = 0.0;

        for (size_t prev = 0; prev < prev_layer->_neurons_size; prev++){
            new_partial_derviative += prev_layer->_weights[prev][n] * prev_layer->_partial_derivatives[prev];
        }
        _partial_derivatives[n] = new_partial_derviative * _derivative(n);
    }

    return this;
}


OLayer* OLayer::calc_output_gradient(std::vector<double>&& expected){

    double error_deriv, deriv_with_error_and_activation;

    for (size_t n = 0; n < _neurons_size; n++){
        error_deriv = 2 * (_activations[n] - expected[n]);

        // Calculate partial derivative: d(cost)/d(activation) * d(activation)/d(weighted_input)
        deriv_with_error_and_activation = error_deriv * _derivative(n);
        _partial_derivatives[n] = deriv_with_error_and_activation;
        // if (std::isnan(deriv_with_error_and_activation)){
        //     while(1){}
        // }
    }
    return this;
}

void OLayer::update_gradients(){
    for (size_t i = 0; i < _neurons_size; i++){

        // partial derivatives are calculated in `calc_output_gradient` and `calc_hidden_gradient`
        double partial_derivative = _partial_derivatives[i];
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
            _gradient_weights[i][j] += partial_derivative * _inputs[j];
        }

        /* 
        Similarily:
            For ouptut layer:
                d(cost)/d(bias) = d(cost)/d(activation) * d(activation)/d(weighted_input) * d(weighted_input)/d(bias)
                                = 2 * (activation - expected) * (derivative of the activation function) * 1
        */
        _gradient_biases[i] += partial_derivative;
    }
}

void OLayer::apply_gradients(double learn_rate, size_t batch_size) {
    double weighted_learn_rate = learn_rate / static_cast<double>(batch_size);

    constexpr double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;

    double gradient;
    for (size_t n = 0; n < _neurons_size; ++n) {
        for(size_t w = 0; w < _inputs_size; ++w) {
            gradient = _gradient_weights[n][w];

            // if (std::isnan(gradient)){
            //     while(1){}
            // }
            _m_gradient[n][w] = beta1 * _m_gradient[n][w] + (1 - beta1) * gradient;
            _v_gradient[n][w] = beta2 * _v_gradient[n][w] + (1 - beta2) * gradient * gradient;
            
            double m_hat = _m_gradient[n][w] / (1.0 - beta1);
            double v_hat = _v_gradient[n][w] / (1.0 - beta2);

            _weights[n][w] -= weighted_learn_rate * m_hat / (sqrt(v_hat) + epsilon);
            _gradient_weights[n][w] = 0.0;
        }
        _biases[n] -= _gradient_biases[n] * weighted_learn_rate;
        _gradient_biases[n] = 0.0;
    }
}

double OLayer::cost(std::vector<double>&& expected) {
    double cost = 0.0;

    for (size_t i = 0; i < _neurons_size; ++i) {
        double error = _activations[i] - expected[i];
        cost += error * error;
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

// returns true if the a ~= b, else (false) a != b.
inline bool isApproximatelyEqual(double a, double b, double epsilon = 0.001){
    return std::abs(a - b) <= epsilon;
}

bool OLayer::operator==(const OLayer& other){
    if (other._neurons_size != _neurons_size || other._inputs_size != _inputs_size){
        return false;
    }

    for (size_t i = 0; i < _neurons_size; i++){
        if (!isApproximatelyEqual(_biases[i], other._biases[i])){
            return false;
        }
        for (size_t j = 0;j < _inputs_size; j++){
            if (!isApproximatelyEqual(_weights[i][j], other._weights[i][j])){
                return false;
            }
        }
    }
    return true;
}

bool OLayer::operator!=(const OLayer& other){
    return !(*this == other);
}

END_NAMESPACE