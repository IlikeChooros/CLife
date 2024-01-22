#include "core/NeuralNetwork.hpp"

START_NAMESPACE_NEURAL_NETWORK

NeuralNetwork::
NeuralNetwork(const std::vector<int>& structure, BaseActivation* activation){
    // First index is number of input neurons
    // Last index is number of output neurons
    // Rest - hidden layers

    if (activation == nullptr){
        activation = new Sigmoid();
    }

    ASSERT(structure.size() >= 2, invalid_structure, "Invalid network sturcture (total size) : " 
        + std::to_string(structure.size()) + " < 2 ")


    _output_layer = new OutputLayer(
        structure.at(structure.size()-2),
        structure.at(structure.size()-1),
        activation
    );

    _hidden_size = structure.size()-2;
    _hidden_layer = new Layer* [_hidden_size];
    
    for (int idx = 1, inputs, outputs; idx < _hidden_size+1; ++idx){
        inputs  = structure.at(idx-1);
        outputs = structure.at(idx);
        _hidden_layer[idx-1] = new Layer(inputs, outputs, activation);
    }
    _structure = structure;

    _curr_loss = 0;
    _average_loss = 0;
    _activation.reset(activation);
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
    assert_data_size(data);
    raw_learn(data);
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
learn(std::vector<data::Data>& batch, size_t apply_batch){
    auto size = batch.size();
    for (size_t i = 0; i < size; i++){
        raw_learn(batch[i]);
        if (i % apply_batch == 0){
            apply(1.1, apply_batch);
            reset_loss();
        }
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
    return _average_loss / batch_size;
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
    double max = INT32_MIN;
    int index = 0;

    for (int i=0; i<_output_layer->node_out; ++i){
        if (_output_layer->outputs[i] > max){
            index = i;
            max = _output_layer->outputs[i];
        }
    }

    return index;
}

// operators

bool NetStructure::operator==(const NetStructure& other){
    
    if (other.size != this->size){
        return false;
    }
    if (other.structure != this->structure){
        return false;
    }

    // now deep compare values
    auto compareDouble = [](double a, double b){
        return std::abs(a) - std::abs(b) < 0.0001;
    };

    auto compareLayer = [&](BaseLayer* layer, BaseLayer* other){
        if (!(layer && other)){
            return false;
        }
        for (int neuronIdx = 0; neuronIdx < layer->node_out; neuronIdx++){
            auto otherNeuron = other->neurons[neuronIdx];
            auto neuron = layer->neurons[neuronIdx];

            if (!compareDouble(otherNeuron->bias, neuron->bias)){
                return false;
            }
            auto size_weights = neuron->weights.size();
            for (size_t w = 0; w < size_weights; w++){
                if (!compareDouble(otherNeuron->weights[w], neuron->weights[w])){
                    return false;
                }
            }
        }
        return true;
    };

    for (int layer = 0; layer < this->size; layer++){
        if (!compareLayer(hidden[layer], other.hidden[layer])){
            return false;
        }
    }

    if (!compareLayer(out, other.out)){
        return false;
    }

    return true;
}

bool NetStructure::operator!=(const NetStructure& other){
    return !this->operator==(other);
}

bool NeuralNetwork::operator==(NeuralNetwork& other){
    std::unique_ptr<NetStructure> str(structure());
    std::unique_ptr<NetStructure> other_str(other.structure());

    return *str == *other_str;
}

bool NeuralNetwork::operator!=(NeuralNetwork& other){
    return !this->operator==(other);
}

bool NeuralNetwork::operator==(const NetStructure& other){
    std::unique_ptr<NetStructure> str(structure());
    return *str == other;
}

bool NeuralNetwork::operator!=(const NetStructure& other){
    return !this->operator==(other);
}

// private

void NeuralNetwork::assert_data_size(const data::Data& data){
    if (data.expect.size() != _output_layer->node_out){
        throw std::runtime_error("NeuralNetwork: Expected data size is not equal to number of output neurons (ex, out): " 
        + std::to_string(data.expect.size()) + " " 
        + std::to_string(_output_layer->node_out));
    }
}

void 
NeuralNetwork::
raw_learn(const data::Data& data){
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

END_NAMESPACE