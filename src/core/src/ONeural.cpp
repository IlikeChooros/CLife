#include <core/ONeural.hpp>

START_NAMESPACE_NEURAL_NETWORK

ONeural::ONeural(
    const std::vector<size_t> &structure,
    ActivationType output_activation,
    ActivationType hidden_activation)
{
    build(structure, output_activation, hidden_activation);
}

ONeural &ONeural::build(
    const std::vector<size_t> &structure,
    ActivationType output_activation,
    ActivationType hidden_activation)
{
    size_t structure_size = structure.size();
    // using macro
    ASSERT(structure_size > 1, invalid_structure,
           "given neural network structure is invalid: " + std::to_string(structure_size) + " expected > 1 (at least with inputs + outputs, no hidden layer)")

    _structure = structure;

    // if there are hidden layers
    if (structure_size > 2)
    {
        size_t hidden_size = structure_size - 2;
        _hidden_layers.reserve(hidden_size);
        for (size_t i = 0; i < hidden_size; i++)
        {
            _hidden_layers.emplace_back(
                structure[i], structure[i + 1],
                std::forward<ActivationType>(hidden_activation));
        }
    }

    auto inputs = structure_size - 2;
    _output_layer.build(
        structure[inputs], structure[inputs + 1],
        std::forward<ActivationType>(output_activation));
    _iterator = 0;
    _cost = 0;
    _loss = 0;
    return *this;
}

void ONeural::initialize()
{
    for (auto &layer : _hidden_layers)
    {
        layer.initialize();
    }
    _output_layer.initialize();
}

void ONeural::_update_gradients(data::Data &&data, ONeural *context)
{
    // This function is made to be thread-safe, it's a static method, because
    // I'm using it in `std::thread` to achieve parallelism

    _NetworkFeedData feed_data(context->_output_layer, context->_hidden_layers);
    context->feed_forward(feed_data, data.input);

    _FeedData *prev_layer_feed = &feed_data._layer_feed_data.back();

    OLayer *prev_layer = context->_output_layer.calc_output_gradient(
        std::forward<vector_t>(data.expect),
        *prev_layer_feed);

    context->_output_layer.update_gradients(*prev_layer_feed);

    {
        // Lock the `_cost` and `_loss` when modifying them
        std::lock_guard<std::mutex> lock(context->_mutex);
        context->_cost = context->_output_layer.cost(
            std::forward<vector_t>(data.expect),
            *prev_layer_feed);
        context->_loss += context->_cost;
    }

    _FeedData *_hidden_feed;
    OLayer *_hidden_layer;

    // backpropagation
    for (int i = (int)context->_hidden_layers.size() - 1; i >= 0; i--)
    {
        _hidden_feed = &feed_data._layer_feed_data[i];
        _hidden_layer = &context->_hidden_layers[i];

        prev_layer = _hidden_layer->calc_hidden_gradient(prev_layer, *_hidden_feed, prev_layer_feed->_partial_derivatives);
        _hidden_layer->update_gradients(*_hidden_feed);
        prev_layer_feed = _hidden_feed;
    }
}

void ONeural::train(data::Data &data)
{
    _update_gradients(std::forward<data::Data>(data), this);
}

void ONeural::learn(data::Data &data, double learn_rate)
{
    _update_gradients(std::forward<data::Data>(data), this);

    apply(learn_rate, 1);
}

void ONeural::_learn_multithread(data_batch *mini_batch, double learn_rate)
{
    ThreadPool pool(std::thread::hardware_concurrency());

    for (size_t i = 0; i < mini_batch->size(); i++)
    {
        pool.enqueue(
            [this, mini_batch, i, learn_rate]()
            {
                _update_gradients(
                    std::forward<data::Data>(mini_batch->at(i)),
                    this);
            });
    }
}

void ONeural::learn(data_batch *training_data, double learn_rate)
{
    // // traning_data should be a copy of the original data, thread safe
    // for (auto& data : *training_data){
    //     _update_gradients(std::forward<data::Data>(data), this);
    // }
    // apply(learn_rate, training_data->size());

    _learn_multithread(training_data, learn_rate);
    apply(learn_rate, training_data->size());
}

void ONeural::batch_learn(data_batch *whole_data, double learn_rate, size_t batch_size)
{
    // divide the data into batch sized chunk
    // end_itr is the end of the batch, prevents from going out of range
    size_t end_itr = std::min((_iterator + 1) * batch_size, whole_data->size());

    _loss = 0;
    data_batch batch(whole_data->begin() + _iterator * batch_size, whole_data->begin() + end_itr);
    learn(&batch, learn_rate);

    _iterator = (end_itr != whole_data->size()) ? _iterator + 1 : 0;
}

void ONeural::apply(double learn_rate, size_t batch_size)
{
    for (auto &layer : _hidden_layers)
    {
        layer.apply_gradients(learn_rate, batch_size);
    }
    _output_layer.apply_gradients(learn_rate, batch_size);
}

void ONeural::feed_forward(_NetworkFeedData &feed_data, vector_t &inputs)
{
    (void)feed_data.setInputs(inputs);

    for (size_t i = 0; i < _hidden_layers.size(); i++)
    {
        feed_data._layer_feed_data[i + 1]._inputs = _hidden_layers[i].calc_activations(feed_data._layer_feed_data[i]);
    }
    _output_layer.calc_activations(feed_data._layer_feed_data.back());
}

vector_t ONeural::outputs()
{
    _NetworkFeedData feed_data(_output_layer, _hidden_layers);
    feed_forward(feed_data, _input.input);
    _outputs = feed_data._layer_feed_data.back()._activations;
    return _outputs;
}

void ONeural::input(data::Data &data)
{
    _input = data;
}

void ONeural::raw_input(const vector_t &_raw_input)
{
    _input.input = _raw_input;
}

real_number_t ONeural::loss(size_t batch_size)
{
    return _loss / static_cast<real_number_t>(batch_size);
}

real_number_t ONeural::cost()
{
    return _cost;
}

size_t ONeural::_classify_feed(_NetworkFeedData &feed_data)
{
    auto output = feed_data._layer_feed_data.back()._activations;
    auto maxElementIterator = std::max_element(
        output.begin(), output.end());
    return std::distance(output.begin(), maxElementIterator);
}

bool ONeural::_correct_feed(_NetworkFeedData &feed_data, vector_t &expected)
{
    return expected[_classify_feed(feed_data)] == 1;
}

size_t ONeural::classify()
{
    _NetworkFeedData feed(_output_layer, _hidden_layers);
    feed._layer_feed_data.back()._activations = outputs();
    return _classify_feed(feed);
}

bool ONeural::correct()
{
    _NetworkFeedData feed(_output_layer, _hidden_layers);
    feed._layer_feed_data.back()._activations = outputs();
    return _correct_feed(feed, _input.expect);
}

const std::vector<size_t> &ONeural::structure()
{
    return _structure;
}

size_t ONeural::_accuracy_multithread(data_batch *mini_test)
{
    ThreadPool pool(std::thread::hardware_concurrency());

    std::mutex mutex;
    size_t correct_count = 0;
    for (size_t i = 0; i < mini_test->size(); i++)
    {
        pool.enqueue(
            [this, mini_test, i, &correct_count, &mutex]()
            {
                _NetworkFeedData feed(_output_layer, _hidden_layers);
                feed_forward(feed, mini_test->at(i).input);
                if (_correct_feed(feed, mini_test->at(i).expect))
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    correct_count++;
                }
            });
    }
    pool.execute();

    return correct_count;
}

real_number_t ONeural::accuracy(data_batch *test)
{
    constexpr size_t batch_size = 32;

    size_t correct_count = 0, begin_itr = 0, end_itr = 0,
           mini_batch_count = test->size() / batch_size,
           remainder = test->size() % batch_size;

    // divide the data into batch sized chunk
    for (size_t i = 0; i < mini_batch_count; i++)
    {
        begin_itr = i * batch_size;
        end_itr = begin_itr + batch_size;
        // This is correct, because in the end, the `_accuracy_multithread(...)` function
        // waits for all the threads to finish, so it's safe to store this data in the stack
        data_batch mini_batch(test->begin() + begin_itr, test->begin() + end_itr);
        correct_count += _accuracy_multithread(&mini_batch);
    }

    // Handle the remainder
    if (remainder != 0)
    {
        data_batch mini_batch(test->begin() + end_itr, test->end());
        correct_count += _accuracy_multithread(&mini_batch);
    }

    return static_cast<real_number_t>(correct_count) / static_cast<real_number_t>(test->size());
}

vector_t ONeural::recall(data_batch *test){
    vector_t recall_matrix(_output_layer._neurons_size, 0);
    vector_t count_matrix(_output_layer._neurons_size, 0);

    ThreadPool pool(std::thread::hardware_concurrency());


    std::mutex mutex;
    for (size_t i = 0; i < test->size(); i++)
    {
        pool.enqueue(
            [this, test, i, &recall_matrix, &count_matrix, &mutex]()
            {
                _NetworkFeedData feed(this->_output_layer, this->_hidden_layers);
                feed_forward(feed, test->at(i).input);
                size_t guess = _classify_feed(feed);
                size_t expect = std::distance(
                    test->at(i).expect.begin(),
                    std::find(test->at(i).expect.begin(), test->at(i).expect.end(), 1)
                );

                std::lock_guard<std::mutex> lock(mutex);
                if (expect == guess){
                    count_matrix[expect]++;
                }
                recall_matrix[expect]++;
            });
    }
    pool.execute();

    for (size_t i = 0; i < recall_matrix.size(); i++)
    {
        recall_matrix[i] = static_cast<real_number_t>(count_matrix[i]) / static_cast<real_number_t>(recall_matrix[i]);
    }

    return recall_matrix;
}

void ONeural::activations(
    ActivationType output_activation,
    ActivationType hidden_activation)
{
    _output_layer._match_activations(output_activation);
    for (auto &layer : _hidden_layers)
    {
        layer._match_activations(hidden_activation);
    }
}

// operators
bool ONeural::operator==(const ONeural &other)
{
    if (_structure != other._structure)
    {
        return false;
    }
    for (size_t i = 0; i < _hidden_layers.size(); i++)
    {
        if (_hidden_layers[i] != other._hidden_layers[i])
        {
            return false;
        }
    }
    if (_output_layer != other._output_layer)
    {
        return false;
    }
    return true;
}

bool ONeural::operator!=(ONeural &other)
{
    return !this->operator==(other);
}

ONeural &ONeural::operator=(const ONeural &other)
{
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