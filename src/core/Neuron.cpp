#include "Neuron.hpp"

BaseNeuron::
BaseNeuron(int conn, BaseActivation* act): _activationFunctor(act){
    construct(conn);
}

BaseNeuron::
BaseNeuron(BaseNeuron* other):
    bias(other->bias), gradient_bias(other->gradient_bias)
{
    weights = other->weights;
    gradient_weights = other->gradient_weights;
}

BaseNeuron::
~BaseNeuron() {}

void
BaseNeuron::
construct(int _conn){
    weights.assign(_conn, 0);
    gradient_weights.assign(_conn, 0);

    gen.prepare(-1.0, 1.0);

    for (int i=0; i<_conn; ++i){
        weights[i] = gen.rand();
    }
    bias = gen.rand();
    gradient_bias = 0;
}

void
BaseNeuron::
create(int conn){
    construct(conn);
}

float fast_sigmoid(float v)
{
    constexpr float c1 = 0.03138777F;
    constexpr float c2 = 0.276281267F;
    constexpr float c_log2f = 1.442695022F;
    v *= c_log2f*0.5;
    int intPart = (int)v;
    float x = (v - intPart);
    float xx = x * x;
    float v1 = c_log2f + c2 * xx;
    float v2 = x + xx * c1 * x;
    float v3 = (v2 + v1);
    *((int*)&v3) += intPart << 24;
    float v4 = v2 - v1;
    float res = v3 / (v3 - v4); 
    return res;
}

void
BaseNeuron::
set_inputs(const std::vector<double>& inputs){
    _inputs = inputs;
}

void 
BaseNeuron::
set_inputs(double* inputs){

}

int
BaseNeuron::
connections(){
    return weights.size();
}

double
BaseNeuron::
activation(){
    double output = bias;
    auto _conn = weights.size();
    for (int i=0; i<_conn; ++i){
        output += weights[i] * _inputs[i];
    }

    _activation = _activationFunctor->activation((float)output);

    return _activation;
}

void 
BaseNeuron::
printNeuron() {
    printf("Neuron: \n \t Biases: %f, %f\n \t Weights: \n \t", bias, gradient_bias);
    
    for (int i = 0; i < weights.size(); i++){
        printf("%f (%f), ", weights[i], gradient_weights[i]);
    }
    printf("\n");
}

double
Neuron::
calculate_gradient(const double& deriv_){
    double node_const = _activation * (1 - _activation) * deriv_;

    gradient_bias += node_const;

    auto _conn = _inputs.size();
    for (int i=0; i<_conn; ++i){
        gradient_weights[i] += _inputs[i] * node_const;
    }

    return node_const;
}

double
OutputNeuron::
calculate_gradient(const double& expected){

    // dc/dw = dc/dA * dA/d(output) * d(output)/dw

    // dc/dA = d (Expected_value - Activation)^2 / d(Activation)
    //       = - 2 * (Expected_value - Activation)

    // dA/d(output) = d( 1 / e^(-output) + 1)/d(output) = A(1 - A)

    // d(output)/dw = d(w*input + bias)/dw = input

    // Final answer dc/dw = - 2 * (Exp - Act) * Act (1 - Act) * input
    // neglecting the '-' because im adding gradient when applying it

    double node_const = 2 * (_activation - expected) * _activationFunctor->derivative(_activation);

    auto _conn = _inputs.size();
    for (int i=0; i<_conn; ++i){
        gradient_weights[i] += _inputs[i] * node_const;
    }

    gradient_bias += node_const;

    return node_const; 
}

void
BaseNeuron::
apply_gradients(const double& learnrate, const int& batch_size){
    bias -= gradient_bias / batch_size * learnrate;
    gradient_bias = 0;

    auto _conn = weights.size();
    for (int i=0; i<_conn; ++i){
        weights[i] -= gradient_weights[i] / batch_size * learnrate;
        gradient_weights[i] = 0;
    }
}   

double
BaseNeuron::
error(const double& expect){
    double error = _activation - expect;
    return error * error;
}
