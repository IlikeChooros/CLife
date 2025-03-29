#include <core/OLayer.hpp>

START_NAMESPACE_NEURAL_NETWORK

_FeedData::_FeedData(size_t inputs, size_t outputs) {
    (void)build(inputs, outputs);
}

_FeedData& _FeedData::build(size_t inputs, size_t outputs){
    _activations = vector_t(outputs, 0);
    _inputs = vector_t(inputs, 0);
    _weighted_inputs = vector_t(outputs, 0);
    _partial_derivatives = vector_t(outputs, 0);
    _dropout_mask = vector_t(outputs, 1);
    return *this;
}

/**
 * For multithreading:
 *  - activations:
 *      - cost()
 *      - derviative():
 *          - calc_activations()
 *          - calc_output_gradient()
 *          - calc_hidden_gradient()
 *  - _inputs
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
 * 
 * Must update functions:
 *  - cost()
 *  
 * 
*/

OLayer::OLayer(size_t inputs, size_t outputs, ActivationType&& type, double dropout){
    (void)build(inputs, outputs, std::forward<ActivationType>(type), dropout);
}

OLayer::OLayer(const OLayer& other){
    (void)(*this = other);
}

OLayer& OLayer::build(size_t inputs, size_t outputs, ActivationType&& type, double dropout){

    _match_activations(type);

    _weights.assign(outputs * inputs, 0);
    _gradient_weights.assign(outputs * inputs, 0);
    _m_gradient.assign(outputs * inputs, 0);
    _v_gradient.assign(outputs * inputs, 0);

    _biases.assign(outputs, 0);
    _gradient_biases.assign(outputs, 0);
    _v_gradient_bias.assign(outputs, 0);
    _m_gradient_bias.assign(outputs, 0);
    
    _error_function.reset(new SquaredError());

    _neurons_size = outputs;
    _inputs_size = inputs;
    _dropout_rate = dropout;
    _calc_outputs_function = _calc_outputs;

    return *this;
}

OLayer& OLayer::initialize(){
    randomize(&_weights, _neurons_size * _inputs_size);
    randomize(&_biases, _neurons_size);

    return *this;
}

void OLayer::training_mode(bool mode){
    if (mode){
        _calc_outputs_function = _calc_outputs_training;
    } else {
        _calc_outputs_function = _calc_outputs;
    }
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

inline vector_t OLayer::_derivative(_FeedData& feed_data){
    switch (_activ_type)
    {
    case ActivationType::silu:
    case ActivationType::selu:
        return _derivative_of_activ(feed_data._weighted_inputs);
    default:
        return _derivative_of_activ(feed_data._activations);
    }
}

vector_t& OLayer::_calc_outputs(OLayer* layer, _FeedData& feed_data){
    // assuming that inputs are already set
    for (size_t i = 0; i < layer->_neurons_size; i++){
        // using equasion:
        // weighted_input = input * weight + bias
        // For mulitple, it is just a sum of all weighted_inputs
        feed_data._weighted_inputs[i] = layer->_biases[i];
        for (size_t j = 0; j < layer->_inputs_size; j++){
            feed_data._weighted_inputs[i] += layer->weight(j, i) * feed_data._inputs[j];
        }
    }
    feed_data._activations = layer->_activation_function(feed_data._weighted_inputs);
    return feed_data._activations;
}

vector_t& OLayer::_calc_outputs_training(OLayer* layer, _FeedData& feed_data){
    // For dropout
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(1.0 - layer->_dropout_rate);

    for (size_t i = 0; i < layer->_neurons_size; i++){
        
        feed_data._dropout_mask[i] = dist(gen);
        feed_data._weighted_inputs[i] = layer->_biases[i] * feed_data._dropout_mask[i];
        for (size_t j = 0; j < layer->_inputs_size; j++){
            feed_data._weighted_inputs[i] += layer->weight(j, i) * feed_data._inputs[j];
        }
    }

    feed_data._activations = layer->_activation_function(feed_data._weighted_inputs);
    return feed_data._activations;
}

vector_t& OLayer::calc_activations(_FeedData& feed_data){
    return _calc_outputs_function(this, feed_data);
}

OLayer* OLayer::calc_hidden_gradient(OLayer* prev_layer, _FeedData& feed_data, vector_t& _prev_partial_derivatives){
    // Outputs should be already calculated: `feed_data._activations`
    // The `prev_layer` is the next layer in the network,
    // so it is the layer that is closer to the output layer.
    // But it is the previous layer in the sense of backpropagation.

    // I'm using the `feed_data` because in order to make the learning process 
    // thread-safe we must move some data to another object.
    // The following values will change every single backpropagation unit:
    // - activations
    // - weighted_inputs
    // - inputs
    // - partial_derivatives
    
    // ***However values below, won't change during learning:
    // - weights
    // - biases
    // - activation_function, derviative
    // - _m_gradient...
    // - _v_gradient...
    // - gradient_weights 
    // - gradient_biases
    //
    // *** -> (as long as the `apply(...)` won't be called)

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
   
    // matrix_t transposed_weights(
    //     prev_layer->_inputs_size, vector_t(prev_layer->_neurons_size, 0)
    // );
    // for (size_t i = 0; i < prev_layer->_inputs_size; i++){
    //     for (size_t j = 0; j < prev_layer->_neurons_size; j++){
    //         transposed_weights[i][j] = prev_layer->_weights[j][i];
    //     }
    // }
    vector_t derviatives(_derivative(feed_data));

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
            new_partial_derviative += prev_layer->weight(n, prev) * _prev_partial_derivatives[prev];
        }
        feed_data._partial_derivatives[n] = new_partial_derviative * derviatives[n] * feed_data._dropout_mask[n];
    }

    return this;
}


OLayer* OLayer::calc_output_gradient(vector_t&& expected, _FeedData& feed_data){
    // Outputs should be already calculated: `feed_data._activations`

    double error_deriv;
    vector_t derviatives(_derivative(feed_data));

    for (size_t n = 0; n < _neurons_size; n++){
        error_deriv = _error_function->derivative(feed_data._activations[n] - expected[n]);

        // Calculate partial derivative: d(cost)/d(activation) * d(activation)/d(weighted_input)
        feed_data._partial_derivatives[n] = error_deriv * derviatives[n];
    }

    return this;
}

void OLayer::update_gradients(_FeedData& feed_data){
    std::lock_guard<std::mutex> lock(_mutex);

    for (size_t i = 0; i < _neurons_size; i++){

        // partial derivatives are calculated in `calc_output_gradient` and `calc_hidden_gradient`
        double partial_derivative = feed_data._partial_derivatives[i];
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
            _gradient_weights[i * _inputs_size + j] += partial_derivative * feed_data._inputs[j];
        }

        /* 
        Similarily:
            For ouptut layer:
                d(cost)/d(bias) = d(cost)/d(activation) * d(activation)/d(weighted_input) * d(weighted_input)/d(bias)
                                = 2 * (activation - expected) * (derivative of the activation function) * 1
                                  <------------------------------------------------------------------->
                                                        partial derivative
        */
        _gradient_biases[i] += partial_derivative;
    }
}

void OLayer::apply_gradients(double learn_rate, size_t batch_size) {
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

    std::lock_guard<std::mutex> lock(_mutex);

    double weighted_learn_rate = learn_rate / static_cast<double>(batch_size);
    real_number_t *gradient_ptr, *_m_gradient_ptr, *_v_gradient_ptr;

    for (size_t n = 0; n < _neurons_size; ++n) {
        for(size_t w = 0; w < _inputs_size; ++w) {
            gradient_ptr = &_gradient_weights[n * _inputs_size + w];
            _m_gradient_ptr = &_m_gradient[n * _inputs_size + w];
            _v_gradient_ptr = &_v_gradient[n * _inputs_size + w];

            *_m_gradient_ptr = beta1 * *_m_gradient_ptr + (1 - beta1) * *gradient_ptr;
            *_v_gradient_ptr = beta2 * *_v_gradient_ptr + (1 - beta2) * *gradient_ptr * *gradient_ptr;
            
            real_number_t m_hat = *_m_gradient_ptr / (1.0 - beta1);
            real_number_t v_hat = *_v_gradient_ptr / (1.0 - beta2);

            _weights[n * _inputs_size + w] -= weighted_learn_rate * m_hat / (sqrt(v_hat) + epsilon);
            *gradient_ptr = 0.0;
        }

        gradient_ptr = &_gradient_biases[n];
        _m_gradient_ptr = &_m_gradient_bias[n];
        _v_gradient_ptr = &_v_gradient_bias[n];

        *_m_gradient_ptr = beta1 * *_m_gradient_ptr + (1 - beta1) * *gradient_ptr;
        *_v_gradient_ptr = beta2 * *_v_gradient_ptr + (1 - beta2) * *gradient_ptr * *gradient_ptr;

        real_number_t m_hat = *_m_gradient_ptr / (1.0 - beta1);
        real_number_t v_hat = *_v_gradient_ptr / (1.0 - beta2);

        _biases[n] -= weighted_learn_rate * m_hat / (sqrt(v_hat) + epsilon);
        *gradient_ptr = 0.0;
    }
}

real_number_t OLayer::cost(vector_t&& expected, _FeedData& feed_data) {
    real_number_t cost = 0.0;

    for (size_t i = 0; i < _neurons_size; ++i) {
        cost += _error_function->output(feed_data._activations[i] - expected[i]);
    }

    return cost;
}

OLayer& OLayer::operator=(const OLayer& other){
    _weights = other._weights;
    _gradient_weights = other._gradient_weights;
    _biases = other._biases;
    _gradient_biases = other._gradient_biases;

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
            if (!isApproximatelyEqual(weight(j, i), other._weights[i * _inputs_size + j])){
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