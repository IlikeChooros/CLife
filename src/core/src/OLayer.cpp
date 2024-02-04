#include <core/OLayer.hpp>

START_NAMESPACE_NEURAL_NETWORK

/**
 * For multithreading:
 *  - activations:
 *      - cost()
 *      - derviative():
 *          - calc_activations()
 *          - calc_output_gradient()
 *          - calc_hidden_gradient()
 *  - _weighted_inputs:
 *      - derviative():
 *          - calc_hidden_gradient()
 *          - calc_output_gradient()
 *  - _partial_derivatives:
 *      - calc_hidden_gradient()
 *      - calc_output_gradient()
 *      - update_gradients()
 *      - apply_gradients()
 *
*/

OLayer::OLayer(size_t inputs, size_t outputs, ActivationType&& type){
    build(inputs, outputs, std::forward<ActivationType>(type));
}

OLayer& OLayer::build(size_t inputs, size_t outputs, ActivationType&& type){

    _match_activations(type);

    _weights.assign(outputs, vector_t(inputs, 0));
    _gradient_weights.assign(outputs, vector_t(inputs, 0));
    _m_gradient.assign(outputs, vector_t(inputs, 0));
    _v_gradient.assign(outputs, vector_t(inputs, 0));

    _biases.assign(outputs, 0);
    _gradient_biases.assign(outputs, 0);
    _activations.assign(outputs, 0);
    _partial_derivatives.assign(outputs, 0);
    _weighted_inputs.assign(outputs, 0);
    _v_gradient_bias.assign(outputs, 0);
    _m_gradient_bias.assign(outputs, 0);
    

    _neurons_size = outputs;
    _inputs_size = inputs;

    return *this;
}

OLayer& OLayer::initialize(){
    std::default_random_engine _engine(std::chrono::system_clock::now().time_since_epoch().count());
    double limit = sqrt(1.0 / _inputs_size);
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

vector_t& OLayer::calc_activations(vector_t&& inputs){
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
   
   /*

   new_partial_derviative += prev_layer->_weights[prev][n] * prev_layer->_partial_derivatives[prev];
   Is quite expensive - a lot of cache misses, a better way would be to store the weights in a transposed matrix.
   Which is what I'm doing here.
   

    Batch 5 loss: 0.461648 time: 523815us
Batch 6 loss: 0.474617 time: 521997us
Batch 7 loss: 0.422769 time: 519369us
Batch 8 loss: 0.517451 time: 513136us
Batch 9 loss: 0.484191 time: 513270us
Batch 10 loss: 0.634847 time: 511389us
Batch 11 loss: 0.619526 time: 513675us
Batch 12 loss: 0.64337 time: 512956us
Batch 13 loss: 0.678748 time: 513511us
Batch 14 loss: 0.424627 time: 512659us
Batch 15 loss: 0.451214 time: 514016us
Batch 16 loss: 0.502156 time: 512944us
Batch 17 loss: 0.591668 time: 513409us
Batch 18 loss: 0.459734 time: 513315us
Batch 19 loss: 0.555463 time: 513417us
Batch 20 loss: 0.526951 time: 512523us
Batch 21 loss: 0.527675 time: 511955us
Batch 22 loss: 0.461886 time: 516144us
Batch 23 loss: 0.491052 time: 514008us
Batch 24 loss: 0.565749 time: 518938us
Batch 25 loss: 0.525129 time: 518238us
Batch 26 loss: 0.573674 time: 515178us
Batch 27 loss: 0.493544 time: 521576us

Without transposed matrix:
    Batch 5 loss: 0.552747 time: 504567us
    Batch 6 loss: 0.499948 time: 502496us
    Batch 7 loss: 0.535107 time: 501914us
    Batch 8 loss: 0.452759 time: 501147us
    Batch 9 loss: 0.48893 time: 503825us
    Batch 10 loss: 0.452703 time: 501794us
    Batch 11 loss: 0.505133 time: 502485us
    Batch 12 loss: 0.534313 time: 502364us
    Batch 13 loss: 0.5288 time: 504265us
    Batch 14 loss: 0.582917 time: 492209us
    Batch 15 loss: 0.49908 time: 489180us
    Batch 16 loss: 0.468601 time: 461509us
    Batch 17 loss: 0.513687 time: 453607us
    Batch 18 loss: 0.455154 time: 451841us
    Batch 19 loss: 0.408404 time: 455022us
    Batch 20 loss: 0.499921 time: 451478us
    Batch 21 loss: 0.604788 time: 452254us
    Batch 22 loss: 0.372709 time: 453725us
    Batch 23 loss: 0.487284 time: 453659us
    Batch 24 loss: 0.434223 time: 454030us


   */

    // matrix_t transposed_weights(
    //     prev_layer->_inputs_size, vector_t(prev_layer->_neurons_size, 0)
    // );
    // for (size_t i = 0; i < prev_layer->_inputs_size; i++){
    //     for (size_t j = 0; j < prev_layer->_neurons_size; j++){
    //         transposed_weights[i][j] = prev_layer->_weights[j][i];
    //     }
    // }

    for (size_t n = 0; n < _neurons_size; n++){

        // auto weigths = std::move(transposed_weights[n]);
        // double new_partial_derviative = std::inner_product(
        //     weigths.begin(),
        //     weigths.end(),
        //     prev_layer->_partial_derivatives.begin(),
        //     0.0
        // );

        double new_partial_derviative = 0.0;
        for (size_t prev = 0; prev < prev_layer->_neurons_size; prev++){
            new_partial_derviative += prev_layer->_weights[prev][n] * prev_layer->_partial_derivatives[prev];
        }
        _partial_derivatives[n] = new_partial_derviative * _derivative(n);
    }

    return this;
}


OLayer* OLayer::calc_output_gradient(vector_t&& expected){

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


    /*
    
    Using ADAM optimizer:

        m_gradient = beta1 * m_gradient + (1 - beta1) * gradient
        v_gradient = beta2 * v_gradient + (1 - beta2) * gradient * gradient

        m_hat = m_gradient / (1 - beta1)
        v_hat = v_gradient / (1 - beta2)

        weight = weight - weighted_learn_rate * m_hat / (sqrt(v_hat) + epsilon)

        with hyperparameters:
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
    
    */
    constexpr double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;

    real_number_t gradient;
    for (size_t n = 0; n < _neurons_size; ++n) {
        for(size_t w = 0; w < _inputs_size; ++w) {
            gradient = _gradient_weights[n][w];

            _m_gradient[n][w] = beta1 * _m_gradient[n][w] + (1 - beta1) * gradient;
            _v_gradient[n][w] = beta2 * _v_gradient[n][w] + (1 - beta2) * gradient * gradient;
            
            real_number_t m_hat = _m_gradient[n][w] / (1.0 - beta1);
            real_number_t v_hat = _v_gradient[n][w] / (1.0 - beta2);

            _weights[n][w] -= weighted_learn_rate * m_hat / (sqrt(v_hat) + epsilon);
            _gradient_weights[n][w] = 0.0;
        }

        gradient = _gradient_biases[n];
        _m_gradient_bias[n] = beta1 * _m_gradient_bias[n] + (1 - beta1) * gradient;
        _v_gradient_bias[n] = beta2 * _v_gradient_bias[n] + (1 - beta2) * gradient * gradient;

        real_number_t m_hat = _m_gradient_bias[n] / (1.0 - beta1);
        real_number_t v_hat = _v_gradient_bias[n] / (1.0 - beta2);

        _biases[n] -= weighted_learn_rate * m_hat / (sqrt(v_hat) + epsilon);
        _gradient_biases[n] = 0.0;
    }
}

real_number_t OLayer::cost(vector_t&& expected) {
    real_number_t cost = 0.0;

    for (size_t i = 0; i < _neurons_size; ++i) {
        real_number_t error = _activations[i] - expected[i];
        cost += error * error;
    }

    return cost;
}

vector_t& OLayer::activations(){
    return _activations;
}

const real_number_t& OLayer::weight(size_t inputIdx, size_t neuronIdx){
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
inline bool isApproximatelyEqual(real_number_t a, real_number_t b, double epsilon = 0.001){
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