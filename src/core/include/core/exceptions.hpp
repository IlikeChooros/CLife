#pragma once

#pragma once

#include <stdexcept>

#include "namespaces.hpp"

START_NAMESPACE_NEURAL_NETWORK

class invalid_structure: public std::exception {
    std::string msg;
    public:
    invalid_structure(const std::string& msg) : msg("invalid_structure: " + msg) {}
    const char* what() {
        return msg.c_str();
    }
};

END_NAMESPACE
