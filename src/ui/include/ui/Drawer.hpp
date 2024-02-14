#pragma once


#include <SFML/Graphics.hpp>
#include <algorithm>
#include <ctime>

#include <core/core.hpp>
#include <backend/backend.hpp>
#include "namespaces.hpp"



START_NAMESPACE_UI

class Drawer{
    sf::RenderWindow window;

    size_t width;
    size_t height;
    size_t pixelRows;
    size_t pixelCols;

    void _draw(sf::Event::MouseButtonEvent& event, bool erase);

    std::vector<std::vector<uint8_t>> _pixels;

    void _drawPixels();

    std::function<void(neural_network::vector_t)> _callback;

    public:
    Drawer(
        size_t width = 512, size_t height = 512,
        size_t pixelRows = 28, size_t pixelCols = 28
    );

    /**
     * @brief Set the callback function, the argument is a 2D vector of doubles - normalized pixels
    */
    Drawer& setCallback(std::function<void(neural_network::vector_t)> callback);

    /**
     * @brief Load image from normalized input `pixels`
    */
    Drawer& loadPixels(const neural_network::vector_t& pixels);

    bool closed();
    void open();
    neural_network::vector_t getPixels();
};

END_NAMESPACE