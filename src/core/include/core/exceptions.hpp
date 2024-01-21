#pragma once

#pragma once

#include <stdexcept>

#include "namespaces.hpp"

#ifndef ASSERT
    #define ASSERT(expression, exceptionType, errMsg) if(!(expression)){throw exceptionType(errMsg);}
#endif

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
