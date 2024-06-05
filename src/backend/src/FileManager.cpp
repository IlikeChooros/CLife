#include "backend/FileManager.hpp"

START_NAMESPACE_BACKEND

namespace fs = std::filesystem;

FileManager::FileManager(
    const std::string &file_name,
    Format format)
{
    prepare(file_name, format);
}

FileManager &FileManager::prepare(
    const std::string &file_name,
    Format format)
{
    _file_name = file_name;
    _format = format;
    prepare_data_dir();

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

void FileManager::prepare_data_dir(){
    if(fs::current_path().filename() == "bin"){
        _path = (fs::current_path().parent_path().parent_path() / "data").string();
    }
    else{
        _path = (fs::current_path() / "data").string();
    }
}

std::vector<std::string> FileManager::list_networks(){
    fs::path path(_path);
    if (!fs::exists(path)){
        std::cout << "Data folder not found!\n";
        throw storage_not_found("Could not open the directory: " + _path);
    }

    std::vector<std::string> networks;
    
    for (const auto &entry : fs::directory_iterator(path)){
        if (entry.is_regular_file()){
            networks.push_back(entry.path().stem().string());
        }
    }
    return networks;
}

void FileManager::_write_binary(
    std::ofstream &writer,
    neural_network::ONeural &network)
{
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

    auto strucutre = network.structure();

    size_t size = strucutre.size();
    writer.write(reinterpret_cast<char *>(&size), sizeof(size));
    for (auto n : strucutre)
    {
        writer.write(reinterpret_cast<char *>(&n), sizeof(n));
        // std::cout << n << ", ";
    }
    std::cout << std::endl;

    using real_number_t = neural_network::real_number_t;

    auto _write_layer = [&writer](neural_network::OLayer& layer){
        real_number_t val;

        for (size_t n = 0; n < layer._neurons_size; n++){
            val = static_cast<real_number_t>(layer._biases[n]);
            writer.write(reinterpret_cast<char *>(&val), sizeof(val));
            // text_file << bias << ", ";
            for (size_t w = 0; w < layer._inputs_size; w++){
                val = static_cast<real_number_t>(layer._weights[n * layer._inputs_size + w]);
                writer.write(reinterpret_cast<char *>(&val), sizeof(val));
                // text_file << weight << ", ";
            }
            // text_file << std::endl;
        }
    };

    // hidden layer
    for (size_t l = 0; l < network._hidden_layers.size(); l++){
        _write_layer(network._hidden_layers[l]);
    }
    
    // output layer
    _write_layer(network._output_layer);

    writer.close();
}

void FileManager::to_file(neural_network::ONeural &network)
{
    std::string fullpath = (fs::path(_path) / (_file_name + _fileFormat)).string();

    switch (_format)
    {
    case Format::binary:{
        std::ofstream _binary_writer(fullpath, std::ios::binary);
        if(!_binary_writer){
            throw storage_not_found("Could not open the file: " + fullpath);
        }
        _write_binary(_binary_writer, network);
    }
        break;

    default:
        break;
    }
}

neural_network::ONeural *FileManager::_read_binary(std::ifstream &reader)
{
    auto structure = _read_structure(reader);
    std::unique_ptr<neural_network::ONeural> net(
        new neural_network::ONeural(structure));
    using real_number_t = neural_network::real_number_t;

    // read hidden layers
    auto _read_layer = [&reader](neural_network::OLayer& layer){
        real_number_t f;

        for (size_t n = 0; n < layer._neurons_size; n++){
            reader.read(reinterpret_cast<char *>(&f), sizeof(f));
            layer._biases[n] = f;

            for (size_t w = 0; w < layer._inputs_size; w++){
                if (reader.eof() || reader.bad()){
                    throw neural_network::invalid_structure(
                        "something went wrong while reading output layer, at: row = " +
                        std::to_string(n) + " col = " + std::to_string(w));
                }

                reader.read(reinterpret_cast<char *>(&f), sizeof(f));
                layer._weights[n * layer._inputs_size + w] = f;
            }
        }
    };
    
    for(size_t l = 0; l < net->_hidden_layers.size(); l++){
        _read_layer(net->_hidden_layers[l]);
    }
    
    // read output layer
    _read_layer(net->_output_layer);

    return net.release();
}

neural_network::ONeural *FileManager::from_file()
{
    std::string fullpath = (fs::path(_path) / (_file_name + _fileFormat)).string();
    
    if (!fs::exists(fullpath))
    {
        throw storage_not_found("Could not open the file: " + fullpath);
    }
    switch (_format)
    {
    case Format::binary:{
        std::ifstream reader(fullpath, std::ios::binary);
        return _read_binary(reader);
    }
    default:
        break;
    }
    return nullptr;
}

std::vector<size_t> FileManager::_read_structure(std::ifstream &reader)
{

    std::vector<size_t> structure;
    size_t size;

    reader.read(reinterpret_cast<char *>(&size), sizeof(size));
    structure.assign(size, 0);

    size_t n;
    for (size_t i = 0; i < size; i++){
        reader.read(reinterpret_cast<char *>(&n), sizeof(n));
        structure[i] = n;
    }

    if (structure.size() < 2 || (!reader.eof() && reader.fail())){
        throw neural_network::invalid_structure(
            "given file: " + _path + " has incorrect "
                                     "neural network structure");
    }
    return structure;
}

END_NAMESPACE