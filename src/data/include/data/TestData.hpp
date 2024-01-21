#pragma once

#include "namespaces.hpp"

#include <vector>
#include "Data.hpp"

START_NAMESPACE_DATA

class TestData{
    public:
    TestData() = default;
    TestData(std::vector<Data>&& batch);
    

    std::vector<Data> batch;
};

END_NAMESPACE