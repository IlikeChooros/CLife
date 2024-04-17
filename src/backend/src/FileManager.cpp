#include "backend/FileManager.hpp"



START_NAMESPACE_BACKEND

FileManager::FileManager(
    const std::string& filePath,
    Format format
) {
    prepare(filePath, format);
}

FileManager& FileManager::prepare(
    const std::string& path, 
    Format format
){
    _path = path;
    _format = format;
    switch (_format)
    {
    case Format::binary:
        _fileFormat = ".bin";
        break;
    case Format::json:
        _fileFormat = ".json";
        break;
    default:
        break;
    }
    return *this;
}

void FileManager::_write_binary(
    std::ofstream& writer, 
    neural_network::ONeural& network
){
    /*
    
    Format:
        (number of inputs) (hidden layers) (number of outputs)
        Ex:
            2 3 4 5 6 2 -> means:
                            - 2 inputs for this network
                            - 3,4,5,6 number of neurons in given hidden layer (total of 4 hidden layers)
                            - 2 output neurons
        (bias + wieghts)
        Ex:
            for structure: 2 3 4 2:
                bias1, wieght1, weight2 <-|
                bias2, wieght1, weight2   | 3 neurons (hidden)
                bias3, wieght1, weight2 <-|
                bias1, wieght1, weight2, weight3 <-|
                bias2, wieght1, weight2, weight3   | 4 neurons (hidden)
                bias3, wieght1, weight2, weight3   | 
                bias4, wieght1, weight2, weight3 <-|
                bias1, wieght1, weight2, weight3, weight4 <-|
                bias1, wieght1, weight2, weight3, weight4 <-| 2 neurons (output neurons)
    
    */

//    std::ofstream text_file("test.txt");

    auto strucutre = network.structure();

    auto size = strucutre.size();
    writer.write(reinterpret_cast<char*>(&size), sizeof(size));
    for (auto n : strucutre){
        writer.write(reinterpret_cast<char*>(&n), sizeof(int));
        // text_file << n << ", ";
    }
    // text_file << std::endl;

    using real_number_t = neural_network::real_number_t;

    // hidden layer
    for (size_t l = 0; l < network._hidden_layers.size(); l++){
        for (size_t n = 0; n < network._hidden_layers[l]._biases.size(); n++){
            real_number_t bias = static_cast<real_number_t>(network._hidden_layers[l]._biases[n]);
            writer.write(reinterpret_cast<char*>(&bias), sizeof(bias));
            // text_file << bias << ", ";
            for (size_t w = 0; w < network._hidden_layers[l]._inputs_size; w++){
                real_number_t weight = static_cast<real_number_t>(network._hidden_layers[l].weight(n, w));
                writer.write(reinterpret_cast<char*>(&weight), sizeof(weight));
                // text_file << weight << ", ";
            }
            // text_file << std::endl;
        }
    }
    // output layer
    for (size_t n = 0; n < network._output_layer._biases.size(); n++){
        real_number_t bias = static_cast<real_number_t>(network._output_layer._biases[n]);
        writer.write(reinterpret_cast<char*>(&bias), sizeof(bias));
        // text_file << bias << ", ";
        for (size_t w = 0; w < network._output_layer._inputs_size; w++){
            real_number_t weight = static_cast<real_number_t>(network._output_layer.weight(w, n));
            writer.write(reinterpret_cast<char*>(&weight), sizeof(weight));
            // text_file << weight << ", ";
        }
        // text_file << std::endl;
    }
    writer.close();
}

void FileManager::to_file(neural_network::ONeural& network){
    std::string fullpath = _path + _fileFormat;
    std::ofstream writer(fullpath);
    if (!writer){
        throw storage_not_found("Could not open the file: " + fullpath);
    }
    switch (_format)
    {
    case Format::binary:
        _write_binary(writer, network);
        break;
    
    default:
        break;
    }
}

neural_network::ONeural* FileManager::_read_binary(std::ifstream& file){
    auto structure = _read_structure(file);
    std::unique_ptr<neural_network::ONeural> net(
        new neural_network::ONeural(structure)
    );
    using real_number_t = neural_network::real_number_t;
    // read hidden layers
    int hidden_size = structure.size() - 2;
    real_number_t f;
    if (hidden_size > 0){
        int size = hidden_size + 1;
        for(int prevLayer = 0, layer = 1; layer < size; ++layer, ++prevLayer ){
            auto neuronsIn = structure[prevLayer];
            auto neuronsOut = structure[layer];

            for (size_t neuron = 0; neuron < neuronsOut; neuron++){
                
                file.read(reinterpret_cast<char*>(&f), sizeof(f));
                net->_hidden_layers[prevLayer]._biases[neuron] = f;
                for (size_t weight = 0; weight < neuronsIn; weight++){
                    if (file.eof() || file.bad()){
                        throw neural_network::invalid_structure(
                            "something went wrong while reading hidden layer, at: row = " +
                            std::to_string(neuron) + " col = " + std::to_string(weight));
                    }
                    file.read(reinterpret_cast<char*>(&f), sizeof(f));
                    net->_hidden_layers[prevLayer]._weights[neuron * neuronsOut + weight] = f;
                }
            }
        }
    }
    // read output layers
    auto neuronsIn = structure[structure.size() - 2];
    auto neuronsOut = structure[structure.size() - 1];

    for (size_t neuron = 0; neuron < neuronsOut; neuron++){
        file.read(reinterpret_cast<char*>(&f), sizeof(f));
        net->_output_layer._biases[neuron] = f;
        for (size_t weight = 0; weight < neuronsIn; weight++){
            if (file.eof() || file.bad()){
                throw neural_network::invalid_structure(
                    "something went wrong while reading output layer, at: row = " +
                    std::to_string(neuron) + " col = " + std::to_string(weight));
            }
            file.read(reinterpret_cast<char*>(&f), sizeof(f));
            net->_output_layer._weights[neuron * neuronsOut + weight] = f;
        }
    }
    return net.release();
}

neural_network::ONeural* FileManager::from_file(){
    std::string fullpath = _path + _fileFormat;
    std::ifstream reader(fullpath, std::ios::binary);
    if (!reader.is_open()){
        throw storage_not_found("Could not open the file: " + fullpath);
    }
    switch (_format)
    {
    case Format::binary:
        return _read_binary(reader);
    default:
        break;
    }
    return nullptr;
}

std::vector<size_t> FileManager::_read_structure(std::ifstream& reader){

    std::vector<size_t> structure;
    size_t size;

    reader.read(reinterpret_cast<char*>(&size), sizeof(size));
    structure.assign(size, 0);

    for (size_t i = 0; i < size; i++){
        int n;
        reader.read(reinterpret_cast<char*>(&n), sizeof(n));
        structure[i] = n;
    }

    if(structure.size() < 2 || (!reader.eof() && reader.fail())){
        throw neural_network::invalid_structure(
            "given file: " + _path + " has incorrect " \
                "neural network structure"
        );
    }
    return structure;
}

END_NAMESPACE