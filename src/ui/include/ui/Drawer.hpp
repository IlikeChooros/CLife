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
    size_t pixelWidth;
    size_t pixelHeight;

    void _draw(sf::Event::MouseButtonEvent& event);

    public:
    Drawer(
        size_t width = 1024, size_t height = 1024,
        size_t pixelsWidth = 28, size_t pixelsHeight = 28
    );

    void open();

};

END_NAMESPACE