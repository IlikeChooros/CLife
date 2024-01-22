#pragma once

#include "namespaces.hpp"

#include <SFML/Graphics.hpp>
#include <algorithm>

#include <core/core.hpp>
#include <test-creator/TestCreator.hpp>
#include <backend/backend.hpp>

START_NAMESPACE_UI

void windowWithDrawer();
void neuralNetworkVisualization(
    sf::RenderWindow& window, neural_network::NeuralNetwork& network
);

void neuralNetworkPointTest();

END_NAMESPACE