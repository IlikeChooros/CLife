#include "test-creator/TestCreator.hpp"

START_NAMESPACE_TEST_CREATOR

TestCreator& TestCreator::prepare(std::function<bool(double,double)> cmp){
    this->_compFunct = cmp;
    return *this;
}

std::vector<data::Data>* 
TestCreator::createPointTest(double min, double max, double itr){
    std::uniform_real_distribution generator(min, max);
    std::default_random_engine engine;

    if ((min > max && itr > 0)){
        std::swap(min, max);
    } else if(min < max && itr < 0){
        itr = std::abs(itr);
    }

    auto testData = new std::vector<data::Data>();
    testData->reserve(std::ceil((max - min) / itr) + 1);
    engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
    while (min <= max){
        std::vector<double> inputs;
        inputs.assign(2, 0);
        inputs[0] = generator(engine);
        inputs[1] = generator(engine);
        bool isCorrect = _compFunct(inputs[0], inputs[1]);
        
        testData->push_back(data::Data(inputs, {(double)isCorrect, (double)!isCorrect}));
        min += itr;
    }

    return testData;
}

END_NAMESPACE