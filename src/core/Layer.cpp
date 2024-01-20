#include "Layer.hpp"

BaseLayer::
BaseLayer(BaseLayer* other):
node_in(other->node_in),
node_out(other->node_out)
{
    neurons = new BaseNeuron*[node_out];

    output_val = other->output_val;
    outputs = other->outputs;
}


BaseLayer::
~BaseLayer(){
    for (int i=0; i<node_out; ++i){
        delete neurons[i];
    }
    delete [] neurons;
}

Layer::
Layer(int inputs, int outputs) {
    create(inputs, outputs);
}

Layer::
Layer(Layer* other):
BaseLayer(other) {
    for (int i = 0; i < node_out; ++i){
        neurons[i] = new Neuron(other->neurons[i]);
    }
}

OutputLayer::
OutputLayer(OutputLayer* other):
BaseLayer(other) {
    for (int i = 0; i < node_out; ++i){
        neurons[i] = new OutputNeuron(other->neurons[i]);
    }
}


void
Layer::
create(int in, int out){
    node_in = in;
    node_out = out;

    neurons = new BaseNeuron* [out];
    output_val.assign(out, 0);
    outputs.assign(out, 0);

    for (int i=0; i<out; ++i){
        neurons[i] = new Neuron(in);
    }
}

OutputLayer::
OutputLayer(int inputs, int outputs) {
    create(inputs, outputs);
}

void
OutputLayer::
create(int in, int out){
    node_in = in;
    node_out = out;

    neurons = new BaseNeuron* [out];
    output_val.assign(out, 0);
    outputs.assign(out, 0);

    for (int i=0; i<out; ++i){
        
        neurons[i] = new OutputNeuron(in);
    }
}

void BaseLayer::
set_inputs(const std::vector<double>& inputs){
    this->inputs = inputs;
}

const std::vector<double>&
BaseLayer::
calc_outputs(){
    for (int i=0; i<node_out; ++i){
        neurons[i]->set_inputs(inputs);
        outputs[i] = neurons[i]->activation();
    }
    return outputs;
}

void
BaseLayer::
apply_gradient(double learnrate = 1.2, int batch_size = 1){
    for (int i=0; i<node_out; ++i){
        neurons[i]->apply_gradients(learnrate, batch_size);
    }
}

Layer*
Layer::
calc_gradient(BaseLayer* prev_layer, const std::vector<double>& node_consts){

    double deriv_;
    for (int i=0; i<node_out; ++i){
        deriv_ = 0;
        for (int loop = 0; loop < prev_layer->node_out; ++loop){
            deriv_ += prev_layer->neurons[loop]->weights[i] * node_consts[loop];
        }
        output_val[i] = neurons[i]->calculate_gradient(deriv_);
    }

    return this;
}

BaseLayer*
OutputLayer::
calc_gradient(const std::vector<double>& expected){
    for (int i=0; i<node_out; ++i){
        output_val[i] = neurons[i]->calculate_gradient(expected.at(i));
    }

    return this;
}

double
OutputLayer::
cost(const std::vector<double>& expected){
    double cost = 0;
    for (int i=0; i<node_out; ++i){
        cost += neurons[i]->error(expected.at(i));
    }
    return cost;
}


