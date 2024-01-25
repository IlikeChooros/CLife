#include <core/ONeural.hpp>

START_NAMESPACE_NEURAL_NETWORK

ONeural::ONeural(const std::vector<double>& structure){
    build(structure);
}

ONeural& ONeural::build(const std::vector<double>& structure){
    // using macro
    ASSERT(structure.size() > 1, invalid_structure, 
        "given neural network structure is invalid: " 
        + std::to_string(structure.size()) 
        + " expected > 1 (at least with inputs + outputs, no hidden layer)"
    )
    
    // if there are hidden layers
    if (structure.size() > 2){
        size_t hidden_size = structure.size() - 2;
        _hidden_layers.reserve(hidden_size);
        for (size_t i = 0; i < hidden_size; i++){
            _hidden_layers.push_back(OLayer(structure[i], structure[i + 1]));
        }
    }

    auto inputs = structure.size() - 2;
    _output_layer.build(structure[inputs], structure[inputs + 1]);
    return *this;
}

ONeural& ONeural::initialize(){
    for (size_t i = 0; i < _hidden_layers.size(); i++){
        _hidden_layers[i].initialize();
    }
    _output_layer.initialize();
    return *this;
}

void ONeural::_update_gradients(data::Data&& data){
    _input = data;

    outputs();
    OLayer* prev_layer = _output_layer.calc_output_gradient(
        std::forward<std::vector<double>>(data.expect)
    );
    _output_layer.update_gradients();

    // backpropagation
    for (int i = int(_hidden_layers.size()) - 1; i >= 0; i--){
        prev_layer = _hidden_layers[i].calc_hidden_gradient(prev_layer);
        _hidden_layers[i].update_gradients();
    }
}

void ONeural::learn(data::Data& data, double learn_rate){
    _update_gradients(std::forward<data::Data>(data));

    for(size_t i = 0; i < _hidden_layers.size(); i++){
        _hidden_layers[i].apply_gradients(learn_rate, 1);
    }
    _output_layer.apply_gradients(learn_rate, 1);
}

void ONeural::learn(data_batch* training_data, double learn_rate){
    for (size_t i = 0; i < training_data->size(); i++){
        _update_gradients(std::forward<data::Data>(training_data->operator[](i)));
    }
    
    for(size_t i = 0; i < _hidden_layers.size(); i++){
        _hidden_layers[i].apply_gradients(learn_rate, training_data->size());
    }
    _output_layer.apply_gradients(learn_rate, training_data->size());
}

const std::vector<double>& ONeural::outputs(){
    auto inputs = _input.input;
    for(size_t i = 0; i < _hidden_layers.size(); i++){
        inputs = _hidden_layers[i].calc_activations(
            std::forward<std::vector<double>>(inputs)
        );
    }
    _outputs = _output_layer.calc_activations(std::forward<std::vector<double>>(inputs));
    return _outputs;
}

void ONeural::input(data::Data& data){
    _input = data;
}

void ONeural::raw_input(const std::vector<double>& _raw_input){
    _input.input = _raw_input;
}

double ONeural::loss(size_t batch_size){
    return 0;
}

double ONeural::cost(){
    return _output_layer.cost(
        std::forward<std::vector<double>>(_input.expect)
        );
}

size_t ONeural::classify(){
    size_t index = 0;
    double max = INT_FAST32_MIN;
    for (size_t i = 0; i < _outputs.size(); i++){
        if(_outputs[i] > max){
            index = i;
        }
    }
    return index;
}

bool ONeural::correct(){
    return _input.expect[classify()] == 1;
}

END_NAMESPACE