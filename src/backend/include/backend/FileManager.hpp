#pragma once

#include <fstream>
#include <memory>
#include <sstream>

#include "namespaces.hpp"
#include "exceptions.hpp"
#include <core/core.hpp>


#define DEBUG_FILE_MANAGER false

#include <iostream>
#include <filesystem>

START_NAMESPACE_BACKEND

enum class Format
{
    binary,
    json
};

class FileManager
{

    std::string _path;
    std::string _file_name;
    Format _format;
    std::string _fileFormat;

    std::vector<size_t> _read_structure(std::ifstream &reader);

    void _write_binary(
        std::ofstream &writer,
        neural_network::ONeural &network);

    neural_network::ONeural *_read_binary(
        std::ifstream &reader);

public:
    FileManager(const std::string &file_name, Format format = Format::binary);
    FileManager() = default;

    FileManager &prepare(
        const std::string &file_name,
        Format format = Format::binary);

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
    void to_file(neural_network::ONeural &network);

    /**
     * @brief Load from file neural network, and save it to the `network`
     * @tparam activation arguments for the ONeural constructor
     */
    neural_network::ONeural *from_file();
};

END_NAMESPACE