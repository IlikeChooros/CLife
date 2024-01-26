#include <core/ONeural.hpp>

START_NAMESPACE_NEURAL_NETWORK

ONeural::ONeural(const std::vector<double>& structure, ActivationType type){
    build(structure, type);
}

ONeural& ONeural::build(const std::vector<double>& structure, ActivationType type){
    // using macro
    ASSERT(structure.size() > 1, invalid_structure, 
        "given neural network structure is invalid: " 
        + std::to_string(structure.size()) 
        + " expected > 1 (at least with inputs + outputs, no hidden layer)"
    )
    
    _structure = structure;
    // if there are hidden layers
    if (structure.size() > 2){
        size_t hidden_size = structure.size() - 2;
        _hidden_layers.reserve(hidden_size);
        for (size_t i = 0; i < hidden_size; i++){
            _hidden_layers.push_back(
                OLayer(
                    structure[i], structure[i + 1], 
                    std::forward<ActivationType>(type))
                );
        }
    }

    auto inputs = structure.size() - 2;
    _output_layer.build(structure[inputs], structure[inputs + 1]);
    _iterator = 0;
    _cost = 0;
    _loss = 0;
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
    _cost = _output_layer.cost(
        std::forward<std::vector<double>>(data.expect)
    );
    _loss += _cost;

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

void ONeural::batch_learn(data_batch* whole_data, double learn_rate, size_t batch_size){
    // divide the data into batch sized chunk
    size_t end_itr = (_iterator + 1)*batch_size;
    size_t start_itr = _iterator * batch_size;

    if (end_itr > whole_data->size()){
        end_itr = whole_data->size();
    }

    _loss = 0;
    data_batch batch(whole_data->begin() + start_itr, whole_data->begin() + end_itr);
    learn(&batch, learn_rate);
    
    if (end_itr != whole_data->size()){
        _iterator++;
    } else{
        _iterator = 0;
    }
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
    return _loss / batch_size;
}

double ONeural::cost(){
    return _output_layer.cost(
        std::forward<std::vector<double>>(_input.expect)
        );
}

size_t ONeural::classify(){
    size_t index = 0;
    double max = INT_LEAST16_MIN;
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

const std::vector<double>& ONeural::structure(){
    return _structure;
}

// operators
bool ONeural::operator==(const ONeural& other){
    if (_structure != other._structure){
        return false;
    }
    for (size_t i = 0; i < _hidden_layers.size(); i++){
        if (_hidden_layers[i] != other._hidden_layers[i]){
            return false;
        }
    }
    if (_output_layer != other._output_layer){
        return false;
    }
    return true;
}

bool ONeural::operator!=(ONeural& other){
    return !this->operator==(other);
}

ONeural& ONeural::operator=(const ONeural& other){
    _structure = other._structure;
    _loss = other._loss;
    _cost = other._cost;
    _iterator = other._iterator;
    _outputs = other._outputs;
    _input = other._input;
    _hidden_layers = other._hidden_layers;
    _output_layer = other._output_layer;
}

END_NAMESPACE