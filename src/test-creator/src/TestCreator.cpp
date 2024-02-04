#include "test-creator/TestCreator.hpp"

START_NAMESPACE_TEST_CREATOR

TestCreator& TestCreator::prepare(std::function<bool(double,double)> cmp){
    this->_compFunct = cmp;
    return *this;
}


data::data_batch* 
TestCreator::createPointTest(double min, double max, size_t size){
    using namespace data;
    std::uniform_real_distribution<real_number_t> generator(min, max);
    std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());

    auto testData = std::make_unique<data_batch>();
    testData->reserve(size);

    for(size_t i = 0; i < size; i++){
        vector_t inputs(2);
        vector_t outputs(2);
        inputs[0] = generator(engine);
        inputs[1] = generator(engine);
        bool isCorrect = _compFunct(inputs[0], inputs[1]);

        outputs[0] = static_cast<real_number_t>(isCorrect);
        outputs[1] = static_cast<real_number_t>(!isCorrect);

        // normalizing the inputs to range <-1,1>
        inputs[0] /= (max - min);
        inputs[1] /= (max - min);
        
        testData->emplace_back(inputs, outputs);
    }

    return testData.release();
}

END_NAMESPACE