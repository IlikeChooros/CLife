#include "backend/FileManager.hpp"

START_NAMESPACE_BACKEND

FileManager::FileManager(
    const std::string& filePath
): _path(filePath) {}

FileManager& FileManager::prepare(const std::string& path){
    _path = path;
    return *this;
}

void FileManager::to_file(neural_network::NeuralNetwork& net){
    std::ofstream writer(_path);

    if (!writer){
        throw storage_not_found("Could not open the file: " + _path);
    }

    /*
    
    Format:
        (number of inputs) (hidden layers) (number of outputs)
        Ex:
            2 3 4 5 6 2 -> means:
                            - 2 inputs for this network
                            - 3,4,5,6 number of neurons in given layer
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
                bias1, wieght1, weight2, weight3, wieght4 <-|
                bias1, wieght1, weight2, weight3, wieght4 <-| 2 neurons (output neurons)
    
    */

    auto strcuture = net.raw_structure();
    for(auto i : strcuture){
        writer << i << " ";
    }
    writer << "\n";

    std::unique_ptr<neural_network::NetStructure> netStruct(net.structure());
    for (int layer = 0; layer < netStruct->size; layer++){

        auto hiddenLayer = netStruct->hidden[layer];

        for(int neuron = 0; neuron < hiddenLayer->node_out; neuron++){

            auto Neuron = hiddenLayer->neurons[neuron];
            writer << Neuron->bias << ' ';
            for (int wieght = 0; wieght < Neuron->weights.size(); wieght++){
                writer << Neuron->weights[wieght] << ' ';
            }
            writer << '\n';
        }
    }
}

neural_network::NeuralNetwork* FileManager::network(){
    std::ifstream reader(_path);

    if (!reader){
        throw storage_not_found("Could not open the file: " + _path);
    }

    std::string line;
    std::getline(reader, line);

    std::vector<int> structure;

    // Read strucutre
    std::stringstream ss(line);
    while (!ss.eof()){
        int n;
        ss >> n;
        structure.push_back(n);
    }

    std::unique_ptr<neural_network::NetStructure> netStruct(
        new neural_network::NetStructure()
    );

    netStruct->structure = structure;
    netStruct->size = structure.size() - 2;

    ASSERT(
        netStruct->size >= 0, neural_network::invalid_structure, 
        "while reading file " + _path + " size of the structure: " 
        + std::to_string(netStruct->size) + " < 2"
    )

    netStruct->hidden = new neural_network::Layer*[netStruct->size];

    for (int prevLayer = 0, layer = 1; layer < netStruct->size + 1; layer++, prevLayer++){

        auto neuronsIn = structure[prevLayer];
        auto neuronsOut = structure[layer];

        auto hiddenLayer = new neural_network::Layer(neuronsIn, neuronsOut);
        netStruct->hidden[prevLayer] = hiddenLayer;

        for (int row = 0; row < neuronsOut; row++){
            auto neuron = hiddenLayer->neurons[row];
            reader >> neuron->bias;
            for(int col = 0; col < neuronsIn; col++){
                reader >> neuron->weights[col];
                if (!reader){
                    throw neural_network::invalid_structure(
                        "FileManager::network(): something went wrong while reading file");
                    break;
                }
            }
        }
    }

    auto neuronsIn = structure[structure.size() - 2];
    auto neuronsOut = structure[structure.size() - 1];

    auto outputLayer = new neural_network::OutputLayer(
        neuronsIn,
        neuronsOut
    );

    netStruct->out = outputLayer;

    neural_network::BaseNeuron **outputNeurons = outputLayer->neurons;
    
    for (int row = 0; row < neuronsOut; row++){
        auto neuron = outputLayer->neurons[row];
        reader >> neuron->bias;
        for(int col = 0; col < neuronsIn; col++){
            reader >> neuron->weights[col];
        }
    }
    
    return new neural_network::NeuralNetwork(netStruct.get());
}

END_NAMESPACE