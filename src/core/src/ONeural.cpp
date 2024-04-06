#include <core/ONeural.hpp>

START_NAMESPACE_NEURAL_NETWORK

ONeural::ONeural(
    const std::vector<size_t>& structure, 
    ActivationType output_activation,
    ActivationType hidden_activation
){
    build(structure, output_activation, hidden_activation);
}

ONeural& ONeural::build(
    const std::vector<size_t>& structure,
    ActivationType output_activation,
    ActivationType hidden_activation
){
    size_t structure_size = structure.size();
    // using macro
    ASSERT(structure_size > 1, invalid_structure, 
        "given neural network structure is invalid: " 
        + std::to_string(structure_size) 
        + " expected > 1 (at least with inputs + outputs, no hidden layer)"
    )
    
    _structure = structure;
    
    // if there are hidden layers
    if (structure_size > 2){
        size_t hidden_size = structure_size - 2;
        _hidden_layers.reserve(hidden_size);
        for (size_t i = 0; i < hidden_size; i++){
            _hidden_layers.emplace_back(
                structure[i], structure[i + 1], 
                std::forward<ActivationType>(hidden_activation)
            );
        }
    }

    auto inputs = structure_size - 2;
    _output_layer.build(
        structure[inputs], structure[inputs + 1], 
        std::forward<ActivationType>(output_activation)
    );
    _iterator = 0;
    _cost = 0;
    _loss = 0;
    return *this;
}

void ONeural::initialize(){
    for(auto& layer : _hidden_layers){
        layer.initialize();
    }
    _output_layer.initialize();
}

void ONeural::_update_gradients_multithread(data::Data&& data, _FeedData& feed_data){

}

void ONeural::_update_gradients(data::Data&& data){
    _input = std::move(data);

    outputs();
    OLayer* prev_layer = _output_layer.calc_output_gradient(
        std::forward<vector_t>(data.expect)
    );
    _output_layer.update_gradients();
    _cost = _output_layer.cost(
        std::forward<vector_t>(data.expect)
    );
    _loss += _cost;

    // backpropagation
    for (auto layer = _hidden_layers.rbegin(); layer != _hidden_layers.rend(); layer++){
        prev_layer = layer->calc_hidden_gradient(prev_layer);
        layer->update_gradients();
    }
}

void ONeural::train(data::Data& data){
    _update_gradients(std::forward<data::Data>(data));
}

void ONeural::learn(data::Data& data, double learn_rate){
    _update_gradients(std::forward<data::Data>(data));

    apply(learn_rate, 1);
}

void ONeural::_learn_multithread(data_batch* training_data, double learn_rate){
    auto size = training_data->size();

    std::thread threads[size];
    
    for (int i = 0; i < size; i++){
        threads[i] = std::thread(
            &ONeural::_update_gradients, this, std::move(training_data->at(i))
        );
    }
    apply(learn_rate, size);
}

void ONeural::learn(data_batch* training_data, double learn_rate){
    // traning_data should be a copy of the original data, thread safe
    for (auto& data : *training_data){
        _update_gradients(std::forward<data::Data>(data));
    }
    apply(learn_rate, training_data->size());
}

void ONeural::batch_learn(data_batch* whole_data, double learn_rate, size_t batch_size){
    // divide the data into batch sized chunk
    // end_itr is the end of the batch, prevents from going out of range
    size_t end_itr = std::min((_iterator + 1) * batch_size, whole_data->size());

    _loss = 0;
    data_batch batch(whole_data->begin() + _iterator * batch_size, whole_data->begin() + end_itr);
    learn(&batch, learn_rate);
    
    _iterator = (end_itr != whole_data->size()) ? _iterator + 1 : 0;
}

void ONeural::apply(double learn_rate, size_t batch_size){
    for(auto& layer : _hidden_layers){
        layer.apply_gradients(learn_rate, batch_size);
    }
    _output_layer.apply_gradients(learn_rate, batch_size);
}

const vector_t& ONeural::outputs(){
    auto inputs = _input.input;
    for(auto& layer : _hidden_layers){
        inputs = layer.calc_activations(
            std::forward<vector_t>(inputs)
        );
    }
    _outputs = _output_layer.calc_activations(
        std::forward<vector_t>(inputs)
    );
    return _outputs;
}



void ONeural::input(data::Data& data){
    _input = data;
}

void ONeural::raw_input(const vector_t& _raw_input){
    _input.input = _raw_input;
}

real_number_t ONeural::loss(size_t batch_size){
    return _loss / static_cast<real_number_t>(batch_size);
}

real_number_t ONeural::cost(){
    return _output_layer.cost(
        std::forward<vector_t>(_input.expect)
        );
}

size_t ONeural::classify() const{
    auto maxElementIterator = std::max_element(
        _outputs.begin(), _outputs.end()
    );
    return std::distance(_outputs.begin(), maxElementIterator);
}

bool ONeural::correct() const{
    return _input.expect[classify()] == 1;
}

const std::vector<size_t>& ONeural::structure(){
    return _structure;
}

real_number_t ONeural::accuracy(data_batch* test){
    size_t correct_count = 0;
    for (auto& data : *test){
        input(data);
        outputs();
        if (correct()){
            correct_count++;
        }
    }
    return static_cast<real_number_t>(correct_count) / static_cast<real_number_t>(test->size());
}

void ONeural::activations(
    ActivationType output_activation,
    ActivationType hidden_activation
){
    _output_layer._match_activations(output_activation);
    for (auto& layer : _hidden_layers){
        layer._match_activations(hidden_activation);
    }
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

    return *this;
}

END_NAMESPACE