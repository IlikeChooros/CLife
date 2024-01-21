#include "test-creator/TestCreator.hpp"

START_NAMESPACE_TEST_CREATOR

TestCreator& TestCreator::prepare(std::function<bool(double,double)> cmp){
    this->_compFunct = cmp;
    return *this;
}

std::vector<data::Data>* 
TestCreator::createPointTest(double min, double max, size_t size){
    std::uniform_real_distribution generator(min, max);
    std::default_random_engine engine;


    auto testData = new std::vector<data::Data>();
    testData->reserve(size);
    engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
    for(size_t i = 0; i < size; i++){
        std::vector<double> inputs;
        inputs.assign(2, 0);
        inputs[0] = generator(engine);
        inputs[1] = generator(engine);
        bool isCorrect = _compFunct(inputs[0], inputs[1]);
        
        testData->push_back(data::Data(inputs, {(double)isCorrect, (double)!isCorrect}));
    }

    return testData;
}

END_NAMESPACE