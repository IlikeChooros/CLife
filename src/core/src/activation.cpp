#include <core/activation.hpp>


START_NAMESPACE_NEURAL_NETWORK


namespace softmax{
    real_number_t activation(vector_t& activations, size_t index){
        vector_t softmax(activations.size());
        real_number_t sum = 10e-6;
        for (size_t i = 0; i < activations.size(); i++){
            softmax[i] = exp(activations[i]);
            sum += softmax[i];
        }
        return softmax[index] / sum;
    }

    real_number_t derivative(vector_t& activations, size_t index){
        return activations[index] * (1 - activations[index]);
    }
}


namespace silu{
    real_number_t activation(vector_t& args, size_t index){
        return args[index] * sigmoid::activation(args, index);
    }
    /// @brief SiLU (Sigmoid Linear Unit), as `derviative` argument - arg, inputs not activations should be passed.
    real_number_t derivative(vector_t& args, size_t index){
        auto sigmoid = sigmoid::activation(args, index);
        return sigmoid * (1 + args[index] * (1 - sigmoid));
    }
}

namespace selu{
    constexpr double 
        alpha = 1.67326, 
        gamma = 1.0507,
        reverse_gamma = 1 / gamma;

    real_number_t activation(vector_t& args, size_t index){
        auto x = args[index];
        if (x > 0){
            return x * gamma;
        }
        return gamma * alpha * (exp(x) - 1);
    }
    // returns act = G * A * Exp(input) - G * A
    // deriv: A * Exp(input) = act / G + A
    real_number_t derivative(vector_t& args, size_t index){
        real_number_t x = args[index];
        if (x > 0){
            return gamma;
        }
        return gamma * alpha * exp(x);
    }
}


namespace prelu
{
    constexpr double aplha = 10e-2;
    real_number_t activation(vector_t& args, size_t index){
        auto x = args[index];
        if (x < 0){
            return aplha*x;
        }
        return x;
    }
    real_number_t derivative(vector_t& activations, size_t index){
        auto x = activations[index];
        if (x < 0){
            return aplha;
        }
        return 1;
    }
}

END_NAMESPACE