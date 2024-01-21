#include "data/TestData.hpp"

START_NAMESPACE_DATA

TestData::TestData(std::vector<Data>&& batch){
    this->batch = batch;
}

END_NAMESPACE