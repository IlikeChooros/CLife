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

    auto strcuture = net.raw_structure();
    for(auto i : strcuture){
        writer << i << " ";
    }
    writer << "\n";

    std::unique_ptr<neural_network::NetStructure> netStruct(net.structure());

    // write hidden layers
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

    // write output layer
    for(int neuron = 0; neuron < netStruct->out->node_out; neuron++){
        auto Neuron = netStruct->out->neurons[neuron];
        writer << Neuron->bias << ' ';
        for (int wieght = 0; wieght < Neuron->weights.size(); wieght++){
            writer << Neuron->weights[wieght] << ' ';
        }
        writer << '\n';
    }
}

neural_network::NeuralNetwork* FileManager::network(){
    std::ifstream reader(_path);

    if (!reader){
        throw storage_not_found("Could not open the file: " + _path);
    }

    try{
        std::string line;
        std::getline(reader, line);

        std::vector<int> structure;

        // Read strucutre
        std::stringstream ss(line);
        while (true){
            int n;
            ss >> n;
    
            if(!(ss.eof() || ss.fail())){
        #if DEBUG_FILE_MANAGER
                std::cout << n << ' ';
        #endif
                structure.push_back(n);
            } else {
                break;
            }
        }

        if(!ss.eof() && ss.bad()){
            throw neural_network::invalid_structure(
                "given file: " + _path + " has incorrect " \
                    "neural network structure"
            );
        }

        std::unique_ptr<neural_network::NetStructure> netStruct(
            new neural_network::NetStructure()
        );

        netStruct->structure = structure;
        // size here, is the size of hidden layers
        netStruct->size = structure.size() - 2;

        ASSERT(
            netStruct->size > 0, neural_network::invalid_structure, 
            "while reading file " + _path + " size of the hidden layer: " 
            + std::to_string(netStruct->size) + " <= 0"
        )

        netStruct->hidden = new neural_network::Layer*[netStruct->size];

        #if DEBUG_FILE_MANAGER
            std::cout << "\nReading hidden layer\n";
        #endif

        for (int prevLayer = 0, layer = 1; layer < netStruct->size + 1; layer++, prevLayer++){

            auto neuronsIn = structure[prevLayer];
            auto neuronsOut = structure[layer];

            auto hiddenLayer = new neural_network::Layer(neuronsIn, neuronsOut);
            netStruct->hidden[prevLayer] = hiddenLayer;

            for (int row = 0; row < neuronsOut; row++){
                auto neuron = hiddenLayer->neurons[row];
                reader >> neuron->bias;

            #if DEBUG_FILE_MANAGER
                std::cout << neuron->bias << ' ';
            #endif

                for(int col = 0; col < neuronsIn; col++){
                    reader >> neuron->weights[col];
                #if DEBUG_FILE_MANAGER
                    std::cout << neuron->weights[col] << ' ';
                #endif
                    if (reader.eof() || reader.fail()){
                        throw neural_network::invalid_structure(
                            "something went wrong while reading hidden layer");
                        break;
                    }
                }

            #if DEBUG_FILE_MANAGER
                std::cout << '\n';
            #endif
            }
        }

        auto neuronsIn = structure[structure.size() - 2];
        auto neuronsOut = structure[structure.size() - 1];

        auto outputLayer = new neural_network::OutputLayer(
            neuronsIn,
            neuronsOut
        );

        netStruct->out = outputLayer;

        #if DEBUG_FILE_MANAGER
            std::cout << "\nReading output layer\n";
        #endif
        
        for (int row = 0; row < neuronsOut; row++){
            auto neuron = outputLayer->neurons[row];
            reader >> neuron->bias;

            #if DEBUG_FILE_MANAGER
                std::cout << neuron->bias << ' ';
            #endif

            for(int col = 0; col < neuronsIn; col++){
                if (reader.eof() || reader.bad()){
                    throw neural_network::invalid_structure(
                        "something went wrong while reading output layer, at: row = " +
                        std::to_string(row) + " col = " + std::to_string(col));
                    break;
                }
                reader >> neuron->weights[col];

                #if DEBUG_FILE_MANAGER
                    std::cout << neuron->weights[col] << ' ';
                #endif
            }

            #if DEBUG_FILE_MANAGER
                std::cout << '\n';
            #endif
        }
        return new neural_network::NeuralNetwork(netStruct.get());
    } catch(neural_network::invalid_structure& err){
        throw std::runtime_error(
            std::string("FileManager::network(): an error occured: ") + err.what());
    } 
    catch(...) {
        throw std::runtime_error(
            "FileManager::network(): something went wrong while reading file");
    }
}

END_NAMESPACE