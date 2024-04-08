#include <core/activation.hpp>


START_NAMESPACE_NEURAL_NETWORK


namespace softmax{
    vector_t activation(vector_t& activations){
        vector_t softmax(activations.size());
        real_number_t sum = 10e-6;
        for (size_t i = 0; i < activations.size(); i++){
            softmax[i] = exp(activations[i]);
            sum += softmax[i];
        }
        for (size_t i = 0; i < activations.size(); i++){
            softmax[i] /= sum;
        }
        return softmax;
    }

    vector_t derivative(vector_t& activations){
        vector_t result(activations.size());
        const size_t size = activations.size();
        for (size_t i = 0; i < size; i++){
            result[i] = activations[i] * (1 - activations[i]);
        }
        return result;
    }
}


namespace silu{
    vector_t activation(vector_t& args){
        vector_t result(args.size());
        const size_t size = args.size();
        auto sigmoid = sigmoid::activation(args);
        for (size_t i = 0; i < size; i++){
            result[i] = args[i] * sigmoid[i];
        }
        return result;
    }
    /// @brief SiLU (Sigmoid Linear Unit), as `derviative` argument - arg, inputs not activations should be passed.
    vector_t derivative(vector_t& args){
        vector_t result(args.size());
        const size_t size = args.size();
        auto sigmoid = sigmoid::activation(args);
        for (size_t i = 0; i < size; i++){
            result[i] = sigmoid[i] * (1 + args[i] * (1 - sigmoid[i]));
        }
        return result;
    }
}

namespace selu{
    constexpr double 
        alpha = 1.67326, 
        gamma = 1.0507,
        reverse_gamma = 1 / gamma;

    vector_t activation(vector_t& args){
        vector_t result(args.size());
        const size_t size = args.size();
        for (size_t i = 0; i < size; i++){
            auto x = args[i];
            if (x > 0){
                result[i] = gamma * x;
            }
            result[i] = gamma * alpha * (exp(x) - 1);
        }
        return result;
    }
    // returns act = G * A * Exp(input) - G * A
    // deriv: A * Exp(input) = act / G + A
    vector_t derivative(vector_t& args){
        vector_t result(args.size());
        const size_t size = args.size();
        for (size_t i = 0; i < size; i++){
            auto x = args[i];
            if (x > 0){
                result[i] = gamma;
            }
            result[i] = gamma * alpha * exp(x);
        }
        return result;
    }
}


namespace prelu
{
    constexpr double aplha = 10e-2;
    vector_t activation(vector_t& args){
        vector_t result(args.size());
        const size_t size = args.size();
        for (size_t i = 0; i < size; i++){
            auto x = args[i];
            if (x < 0){
                result[i] = aplha*x;
            }
            result[i] = x;
        }
        return result;
    }
    vector_t derivative(vector_t& activations){
        vector_t result(activations.size());
        const size_t size = activations.size();
        for (size_t i = 0; i < size; i++){
            auto x = activations[i];
            if (x < 0){
                result[i] = aplha;
            }
            result[i] = 1;
        }
        return result;
    }
}

END_NAMESPACE