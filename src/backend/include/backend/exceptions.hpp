#pragma once

#include <stdexcept>

#include "namespaces.hpp"

#ifndef ASSERT
    #define ASSERT(expression, exceptionType, errMsg) if(!(expression)){throw exceptionType(errMsg);}
#endif

START_NAMESPACE_BACKEND

class storage_not_found: public std::exception {
    std::string msg;
    public:
    storage_not_found(const std::string& msg) : msg("storage_not_found: " + msg) {}
    const char* what() {
        return msg.c_str();
    }
};


END_NAMESPACE
