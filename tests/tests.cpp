#include "tests.hpp"

START_NAMESPACE_TESTS

    void root()
    {
        using testBase = std::shared_ptr<TestCase>;
        std::vector<testBase> tests = {
            testBase(new SimpleNeuronTest()),
            testBase(new SimpleLayerTest()),
            testBase(new SimpleNeuralNetworkTest()),
            testBase(new SaveNeuralNetworkToFile()),
        };

        int failed = 0;

        for (auto test : tests)
        {
            test->run();
            if (test->failed())
            {
                failed++;
            }
        }
        printf(
            "\n********** SUMMARY **********\n\t"
                "Total of tests: %lu\n\t"
                "Failed: %i\n",
            tests.size(),
            failed
        );
    }
END_NAMESPACE
