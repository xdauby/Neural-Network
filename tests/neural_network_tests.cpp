#include <stdio.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include "data_set.hpp"
#include "edge.hpp"
#include "neural_network.hpp"
#include "node.hpp"
#include "loss_type.hpp"
#include "activation_type.hpp"

/**
 * @brief compute relative Error between to vectors
 *
 * @param x_hat
 * @param x
 * @return the error
 */
double relativeError(std::vector<double> x_hat, std::vector<double> x) {
    if (x_hat.size() != x.size()) {
        throw std::invalid_argument("x_hat size's different from x size's !");
    }

    double epsilon = 0.0000000000001;
    double x_hat_minus_x_norm = 0;
    double x_norm = 0;

    for (unsigned int index{0}; index < x_hat.size(); index++) {
        x_hat_minus_x_norm += std::abs(x_hat[index] - x[index]);
        x_norm += std::abs(x[index]);
    }
    return x_hat_minus_x_norm / (x_norm + epsilon);
}

/**
 * @brief test the number of weights
 *
 */
void weightsNumberTest() {
    int inputLayerSize{4};
    std::vector<int> hiddenLayerSizes{3, 2};
    int outputLayerSize{2};
    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, ActivationType::LEAKYRELU, ActivationType::SIGMOID,
                     LossType::NONE);
    // expected number of weights is 29 : 4*3 + 3 + 3*2 + 2 + 2*2 + 2 = 33
    unsigned int expectednWeights = 29;
    unsigned int nWeights = nn.getnWeights();

    assert(expectednWeights == nWeights);
}

/**
 * @brief test getWeights method
 *
 */
void getWeightsTest() {
    int inputLayerSize{2};
    std::vector<int> hiddenLayerSizes{2, 2};
    int outputLayerSize{1};
    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, ActivationType::LEAKYRELU, ActivationType::SIGMOID,
                     LossType::NONE);

    std::vector<double> expectedWeights = {0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1};
    std::vector<double> weights = nn.getWeights();
    assert(expectedWeights == weights);
}

/**
 * @brief test setWeights method
 *
 */
void setWeightsTest() {
    int inputLayerSize{2};
    std::vector<int> hiddenLayerSizes{2, 2};
    int outputLayerSize{1};
    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, ActivationType::LEAKYRELU, ActivationType::SIGMOID,
                     LossType::NONE);

    std::vector<double> expectedWeights = {-0.25, -1, 1, 0.5, 1.5, 1.5, 0, 8, -5, 0, 1, 3, 0, 1, 1};
    nn.setWeights(expectedWeights);
    std::vector<double> weights = nn.getWeights();

    assert(expectedWeights == weights);
}

/**
 * @brief test forwardPropagation method for loss function "None"
 *
 */
void forwardPropagationTest() {
    int inputLayerSize{2};
    std::vector<int> hiddenLayerSizes{2};
    int outputLayerSize{1};

    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, ActivationType::SIGMOID, ActivationType::SIGMOID,
                     LossType::NONE);

    std::vector<double> inputData = {1.5, 0.5};
    std::vector<double> inputLabels = {5};
    double output = nn.forwardPropagation(inputData, inputLabels);
    double expectedOutput = 0.853409204;

    assert(std::abs(output - expectedOutput) < 1e-7);
}

/**
 * @brief test forwardPropagation method for loss function "L2 norm"
 *
 */
void forwardPropagationL2Test() {
    int inputLayerSize{2};
    std::vector<int> hiddenLayerSizes{2};
    int outputLayerSize{2};

    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, ActivationType::SIGMOID, ActivationType::SIGMOID,
                     LossType::L2NORM);

    std::vector<double> inputData = {1.5, 0.5};
    std::vector<double> inputLabels = {2, 3};
    double output = nn.forwardPropagation(inputData, inputLabels);
    double expectedOutput = 2.433623327;

    assert(std::abs(output - expectedOutput) < 1e-7);
}

/**
 * @brief test getNumericGradient method
 *
 */
void numericGradientTest1() {
    int inputLayerSize{2};
    std::vector<int> hiddenLayerSizes{2};
    int outputLayerSize{2};

    // sigmoid was used in hidden layer just for the test
    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, ActivationType::SIGMOID, ActivationType::SIGMOID,
                     LossType::NONE);

    std::vector<double> inputData = {1.5, 0.5};
    std::vector<double> inputLabels = {2, 2};
    std::vector<double> expectedGradient = {0.0262698012, 0.0394047018, 0.0131349006, 0.0262698012,
                                            0.0394047018, 0.0131349006, 0.1251019341, 0.110189418,
                                            0.110189418,  0.1251019341, 0.110189418,  0.110189418};

    std::vector<double> gradient = nn.getNumericGradient(inputData, inputLabels);
    assert(relativeError(gradient, expectedGradient) < 1e-6);
}

/**
 * @brief test getNumericGradient method
 *
 */
void numericGradientTest2() {
    int inputLayerSize{2};
    std::vector<int> hiddenLayerSizes{2};
    int outputLayerSize{2};

    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, ActivationType::LEAKYRELU, ActivationType::LINEAR,
                     LossType::L2NORM);

    nn.setWeights(std::vector<double>{0, -1, 1, 0, 0.5, 0.5, 0, 1, 1, 0, 0.5, -0.5});

    std::vector<double> inputData = {2, 1};
    std::vector<double> inputLabels = {2, 3};
    std::vector<double> expectedGradient = {-0.015835, -0.0316699, -0.015835, 0.357006,
                                            0.714013,  0.357006,   -0.138196, 0.0034549,
                                            -0.207294, -0.990405,  0.0247601, -1.4856074};

    std::vector<double> gradient = nn.getNumericGradient(inputData, inputLabels);
    assert(relativeError(gradient, expectedGradient) < 1e-6);
}

/**
 * @brief test get gradient method for loss function "None"
 *
 */
void backwardPropagationTest() {
    int inputLayerSize{2};
    std::vector<int> hiddenLayerSizes{2};
    int outputLayerSize{2};

    // sigmoid was used in hidden layer just for the test
    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, ActivationType::SIGMOID, ActivationType::SIGMOID,
                     LossType::NONE);

    std::vector<double> inputData = {0.096, 0.5};
    std::vector<double> inputLabels = {6, 2};

    nn.forwardPropagation(inputData, inputLabels);
    nn.backwardPropagation();
    std::vector<double> deltas = nn.getDeltas();
    std::vector<double> gradient = nn.getNumericGradient(inputData, inputLabels);

    assert(relativeError(deltas, gradient) < 1e-6);
}

/**
 * @brief test get gradient method for loss function "L2 norm"
 *
 */
void backwardPropagationTestL2() {
    int inputLayerSize{2};
    std::vector<int> hiddenLayerSizes{2};
    int outputLayerSize{2};

    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, ActivationType::LEAKYRELU, ActivationType::LINEAR,
                     LossType::L2NORM);

    nn.setWeights(std::vector<double>{0, -1, 1, 0, 0.5, 0.5, 0, 1, 1, 0, 0.5, -0.5});

    std::vector<double> inputData = {2.5, -1};
    std::vector<double> inputLabels = {2, 3};

    nn.forwardPropagation(inputData, inputLabels);
    nn.backwardPropagation();
    std::vector<double> deltas = nn.getDeltas();
    std::vector<double> gradient = nn.getNumericGradient(inputData, inputLabels);

    assert(relativeError(deltas, gradient) < 1e-6);
}

/**
 * @brief test get gradient method for loss function "Softmax"
 *
 */
void backwardPropagationTestSoftmax() {
    int inputLayerSize{2};
    std::vector<int> hiddenLayerSizes{2};
    int outputLayerSize{2};

    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, ActivationType::LEAKYRELU, ActivationType::LINEAR,
                     LossType::SOFTMAX);

    nn.setWeights(std::vector<double>{0, -1, 1, 0, 0.5, 0.5, 0, 1, 1, 0, 0.5, -0.5});

    std::vector<double> inputData = {2.5, -1};
    std::vector<double> inputLabels = {1};

    nn.forwardPropagation(inputData, inputLabels);
    nn.backwardPropagation();
    std::vector<double> deltas = nn.getDeltas();
    std::vector<double> gradient = nn.getNumericGradient(inputData, inputLabels);

    assert(relativeError(deltas, gradient) < 1e-6);
}

/**
 * @brief test to import dataset
 *
 */
void importDataSetTest() {
    DataSet dataSetTest("./tests/datasets/dataset_data_test.csv", "./tests/datasets/dataset_labels_test.csv", "classification");
    std::vector<std::vector<double>> inputData = dataSetTest.getInputData();
    std::vector<std::vector<double>> inputLabels = dataSetTest.getInputLabels();

    std::vector<std::vector<double>> expectedInputData = {
        {1, 2.1, 2}, {3, 1, 2}, {1, 0, 0}, {10.2, 9, 8}};

    std::vector<std::vector<double>> expectedInputLabels = {{1}, {2}, {0}, {1}};
    assert(inputData == expectedInputData);
    assert(inputLabels == expectedInputLabels);
}

int main() {
    weightsNumberTest();
    getWeightsTest();
    setWeightsTest();
    forwardPropagationTest();
    forwardPropagationL2Test();
    numericGradientTest1();
    numericGradientTest2();
    backwardPropagationTest();
    backwardPropagationTestL2();
    backwardPropagationTestSoftmax();
    importDataSetTest();
}
