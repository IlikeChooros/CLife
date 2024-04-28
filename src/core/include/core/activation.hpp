#pragma once

#include <algorithm>
#include <cmath>

#include "namespaces.hpp"
#include "types.hpp"

enum class ActivationType {
    sigmoid,
    relu,
    softmax,
    silu,
    selu,
    prelu
};

enum class ErrorFunctionType {
    squared_error,
    cross_entropy
};

class BaseErrorFunction{
public:
    BaseErrorFunction(ErrorFunctionType&& type): type(type) {}
    virtual double output(double arg) = 0;
    virtual double derivative(double error) = 0;
    ErrorFunctionType type;
};

class SquaredError: public BaseErrorFunction{
public:
    SquaredError() : BaseErrorFunction(ErrorFunctionType::squared_error) {}
    double output(double arg) override{
        return 0.5 * arg * arg;
    }
    double derivative(double error) override{
        return error;
    }
};


/// @brief Cross-entropy error function - use it only with softmax activation function
class CrossEntropy: public BaseErrorFunction{
public:
    CrossEntropy() : BaseErrorFunction(ErrorFunctionType::cross_entropy) {}
    double output(double arg) override{
        return -log(arg + 1e-15);
    }
    double derivative(double error) override{
        return -1.0 / error;
    }
};

/// New - namespace approach
namespace neural_network{
    namespace sigmoid{
        inline vector_t activation(vector_t& activations){
            vector_t result(activations.size());
            const size_t size = activations.size();
            for (size_t i = 0; i < size; i++){
                result[i] = 1 / (1 + exp(-activations[i]));
            }
            return result;
        }
        inline vector_t derivative(vector_t& activations){
            vector_t result(activations.size());
            const size_t size = activations.size();
            for (size_t i = 0; i < size; i++){
                result[i] = activations[i] * (1 - activations[i]);
            }
            return result;
        }
    }
    namespace relu
    {
        inline vector_t activation(vector_t& activations){
            vector_t result(activations.size());
            const size_t size = activations.size();
            for (size_t i = 0; i < size; i++){
                result[i] = std::max((real_number_t)0, activations[i]);
            }
            return result;
        }
        inline vector_t derivative(vector_t& activations){
            vector_t result(activations.size());
            const size_t size = activations.size();
            for (size_t i = 0; i < size; i++){
                result[i] = activations[i] > 0 ? 1 : 0;
            }
            return result;
        }
    } // namespace relu

    namespace softmax
    {
        vector_t activation(vector_t& args);
        vector_t derivative(vector_t& activations);
    }

    namespace silu
    {
        vector_t activation(vector_t& args);
        vector_t derivative(vector_t& activations);
    }

    namespace selu
    {
        vector_t activation(vector_t& args);
        vector_t derivative(vector_t& activations);
    }
    namespace prelu
    {
        vector_t activation(vector_t& args);
        vector_t derivative(vector_t& activations);
    }
}