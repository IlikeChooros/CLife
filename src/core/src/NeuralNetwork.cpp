#include "core/NeuralNetwork.hpp"

START_NAMESPACE_NEURAL_NETWORK

NeuralNetwork::
NeuralNetwork(const std::vector<int>& structure){
    // First index is number of input neurons
    // Last index is number of output neurons
    // Rest - hidden layers

    // if (structure.size() < 2){
    //     throw invalid_structure("Invalid network sturcture (total size) : "
    //     + std::to_string(structure.size()) + " < 2 " );
    // }

    ASSERT(structure.size() >= 2, invalid_structure, "Invalid network sturcture (total size) : " 
        + std::to_string(structure.size()) + " < 2 ")


    _output_layer = new OutputLayer(structure.at(structure.size()-2), structure.at(structure.size()-1));

    _hidden_size = structure.size()-2;
    _hidden_layer = new Layer* [_hidden_size];
    
    for (int idx = 1, inputs, outputs; idx < _hidden_size+1; ++idx){
        inputs  = structure.at(idx-1);
        outputs = structure.at(idx);
        _hidden_layer[idx-1] = new Layer(inputs, outputs);
    }
    _structure = structure;
}

NeuralNetwork::
NeuralNetwork(
    NetStructure* st
): _structure(st->structure), _hidden_size(st->size)
{
    _hidden_layer = new Layer* [_hidden_size];
    for (int i = 0; i < _hidden_size; ++i){
        _hidden_layer[i] = new Layer(st->hidden[i]);
    }
    _output_layer = new OutputLayer(st->out);
}

NeuralNetwork::
~NeuralNetwork(){
    delete _output_layer;
    for (int i = 0; i < _hidden_size; ++i){
        delete _hidden_layer[i];
    }
    delete [] _hidden_layer;
}

[[nodiscard]]
NetStructure*
NeuralNetwork::
structure(){
    NetStructure* str = new NetStructure;

    str->structure = _structure;
    str->size = _hidden_size;
    str->hidden = new Layer* [str->size];

    for (int i=0; i < str->size; ++i){
        str->hidden[i] = new Layer(_hidden_layer[i]);
    }
    str->out = new OutputLayer(_output_layer);

    return str;
}

void
NeuralNetwork::
learn(data::Data& data){
    if (data.expect.size() != _output_layer->node_out){
        throw std::runtime_error("NeuralNetwork: Expected data size is not equal to number of output neurons (ex, out): " 
        + std::to_string(data.expect.size()) + " " 
        + std::to_string(_output_layer->node_out));
    }
    
    input = data;

    output();

    BaseLayer* prev_layer;
    std::vector<double> output_val;

    prev_layer = _output_layer->calc_gradient(data.expect);
    output_val = _output_layer->output_val;

    _curr_loss = _output_layer->cost(data.expect);
    _average_loss += _curr_loss; 

    // Backpropagation
    for (int reverse = _hidden_size-1; reverse > -1; --reverse){
        prev_layer = _hidden_layer[reverse]->calc_gradient(prev_layer, output_val);
        output_val = prev_layer->output_val;
    }
}

void
NeuralNetwork::
set_input(data::Data& data){
    input = data;
}

void
NeuralNetwork::
raw_input(const std::vector<double>& inputs){
    this->input.input = inputs;
}


void
NeuralNetwork::
learn(const std::vector<data::Data>& batch){
    for (auto data : batch){
        learn(data);
    }
}

void
NeuralNetwork::
apply(double learn_rate, int batch_size){
    for (int i=0; i<_hidden_size; ++i){
        _hidden_layer[i]->apply_gradient(learn_rate, batch_size);
    }
}

double
NeuralNetwork::
cost(){
    return _curr_loss;
}

std::vector<double>
NeuralNetwork::
get_outputs(){
    return std::vector<double>(_output_layer->outputs);
}

double
NeuralNetwork::
loss(int batch_size){
    double ret = _average_loss / batch_size;
    return ret;
}

void 
NeuralNetwork::
reset_loss(){
    _average_loss = 0;
}

bool
NeuralNetwork::
correct(){
    return input.expect[classify()] == 1;
}

void
NeuralNetwork::
output(){
    auto temp = input.input;

    for (int i=0; i<_hidden_size; ++i){
        _hidden_layer[i]->inputs = temp;
        temp = _hidden_layer[i]->calc_outputs();
    }

    _output_layer->inputs = temp;
    _output_layer->calc_outputs();
}

int 
NeuralNetwork::
classify(){
    double max = -1;
    int index = 0;

    for (int i=0; i<_output_layer->node_out; ++i){
        if (_output_layer->outputs[i] > max){
            index = i;
            max = _output_layer->outputs[i];
        }
    }

    return index;
}


END_NAMESPACE