#pragma once

#include "namespaces.hpp"
#include "exceptions.hpp"
#include <core/core.hpp>

#include <fstream>
#include <memory>
#include <sstream>


START_NAMESPACE_BACKEND

class FileManager{

    std::string _path;
    public:
    FileManager(const std::string& filepath);
    FileManager() = default;

    FileManager& prepare(const std::string& path);

    
    /*
    stores neural network to the file, using format (with example):
        - structure (2 3 4 2)
        - bias + weights:
            1. (-1 0.5 0.1) 
            1. (-0.5 1 0.7)                   
            1. (-0.5 0.8 0.5)               
                ^------------^ 1 + 2
            1. (0.5 1 0.25 0.5)
            1. (0.7 0.5 0.37 0.54)
            1. (-0.3 0.7 0.25 0.7)
            1. (0.1 0.2 0.6 -0.4)
                ^-----------------^ 1 + 3
            1. (0.5 1 0.25 0.5 0.7)
            1. (0.7 0.5 0.37 0.54 -0.1)
                ^-------------------^ 1 + 4
    */
    void to_file(neural_network::NeuralNetwork& network);
    neural_network::NeuralNetwork* network();
};

END_NAMESPACE