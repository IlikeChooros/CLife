#include "TestCase.hpp"

size_t micros(){
    return std::chrono::system_clock::now().time_since_epoch().count() / 1000;
}

START_NAMESPACE_TESTS

    TestCase::TestCase(std::string name):
     name(name), _failed(false) {}

    void TestCase::run(){
        _testWrapper();
    }

    bool TestCase::failed(){
        return _failed;
    }

    // private / protected methods

    void TestCase::_successfulPass(){
        printf("^^^^^^^^^^^^Successfuly passed %s ^^^^^^^^^^^^\n", name.c_str());
    }

    void TestCase::_onFail(const std::runtime_error& err){
        printf("XXXXXX Test failed: %s XXXXXX\n\t Reason: %s\n", name.c_str(), err.what());
    }

    void TestCase::_startTest(){
        printf("\n--------------Testing: %s --------------\n", name.c_str());
    }

    void TestCase::_report(int timeNs, int timeMs, int memoryDiff){
        printf("\n***Report***\n\tFinished in: %i ns ( %i ms)\n\tMemory difference: %i B\n",
                timeNs, timeMs, memoryDiff);
    }

    void TestCase::_testWrapper(){
        _startTest();

        try{
            // auto memoryStart = ESP.getFreeHeap();
            auto startNs = micros();
            test();
            auto timeNs = micros() - startNs;
            _report(timeNs, timeNs / 1000, 0);
            _successfulPass();
        } catch(const std::runtime_error& e){
            _onFail(e);
            _failed = true;
        }
    }

    void TestCase::_assert(bool c, std::string assertName){
        if (!c){
            throw std::runtime_error("Assert failed: " + assertName);
        }
    }

    void TestCase::assertTrue(bool c){
        _assert(c, "assertTrue");
    }

    void TestCase::assertFalse(bool c){
        _assert(!c, "assertFalse");
    }

    void TestCase::assertThrow(std::function<void(void)> throableFunction){
        bool threw = false;
        __try {
            throableFunction();
        } __catch(...) {
            threw = true;
        }
        _assert(threw, "assertThrow");
    }

END_NAMESPACE