#include <iostream>
#include <vector>
#include <cassert>
#include <stdio.h>
#include <cmath>
#include "neural_network.hpp"
#include "edge.hpp"
#include "node.hpp"
#include "data_set.hpp"

double relativeError(std::vector<double> x_hat, std::vector<double> x)
{
    if (x_hat.size() != x.size()){
        throw std::invalid_argument("x_hat size's different from x size's !");
    }

    double epsilon = 0.0000000000001;
    double x_hat_minus_x_norm = 0;
    double x_norm = 0;

    for(unsigned int index {0}; index < x_hat.size(); index++){
        x_hat_minus_x_norm += std::abs(x_hat[index] - x[index]);
        x_norm += std::abs(x[index]);
    }
    return x_hat_minus_x_norm / (x_norm + epsilon);
    
}

void weightsNumberTest()
{
    int inputLayerSize {4};
    std::vector<int> hiddenLayerSizes {3,2};
    int outputLayerSize {2};
    std::string lossFunction = "None";
    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, "leakyrelu", "sigmoid", lossFunction);    
    //expected number of weights is 29 : 4*3 + 3 + 3*2 + 2 + 2*2 + 2 = 33
    unsigned int expectednWeights = 29;
    unsigned int nWeights = nn.getnWeights();
    
    assert(expectednWeights == nWeights);
}

void getWeightsTest()
{
    int inputLayerSize {2};
    std::vector<int> hiddenLayerSizes {2,2};
    int outputLayerSize {1};
    std::string lossFunction = "None";
    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, "leakyrelu", "sigmoid", lossFunction);    

    std::vector<double> expectedWeights = {0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1};
    std::vector<double> weights = nn.getWeights();
    assert(expectedWeights == weights);
}

void setWeightsTest()
{
    int inputLayerSize {2};
    std::vector<int> hiddenLayerSizes {2,2};
    int outputLayerSize {1};
    std::string lossFunction = "None";
    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, "leakyrelu", "sigmoid", lossFunction);    

    std::vector<double> expectedWeights = {-0.25, -1, 1, 0.5, 1.5, 1.5, 0, 8, -5, 0, 1, 3, 0, 1, 1};
    nn.setWeights(expectedWeights);
    std::vector<double> weights = nn.getWeights();
    
    assert(expectedWeights == weights);
}

void forwardPropagationTest()
{

    int inputLayerSize {2};
    std::vector<int> hiddenLayerSizes {2};
    int outputLayerSize {1};
    std::string lossFunction = "None";
    
    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, "sigmoid", "sigmoid", lossFunction); 
    
    std::vector<double> inputData = {1.5, 0.5};
    std::vector<double> inputLabels = {5};    
    double output = nn.forwardPropagation(inputData, inputLabels);
    double expectedOutput = 0.853409204;

    assert(std::abs(output - expectedOutput) < 1e-7);

}

void forwardPropagationL2Test()
{

    int inputLayerSize {2};
    std::vector<int> hiddenLayerSizes {2};
    int outputLayerSize {2};
    std::string lossFunction = "L2 norm";

    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, "sigmoid", "sigmoid", lossFunction); 
    
    std::vector<double> inputData = {1.5, 0.5};
    std::vector<double> inputLabels = {2, 3};    
    double output = nn.forwardPropagation(inputData, inputLabels);
    double expectedOutput = 2.433623327;

    assert(std::abs(output - expectedOutput) < 1e-7);

}

void numericGradientTest1()
{
    int inputLayerSize {2};
    std::vector<int> hiddenLayerSizes {2};
    int outputLayerSize {2};
    std::string lossFunction = "None";

    //sigmoid was used in hidden layer just for the test
    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, "sigmoid", "sigmoid", lossFunction); 
    
    std::vector<double> inputData = {1.5, 0.5};
    std::vector<double> inputLabels = {2, 2};
    std::vector<double> expectedGradient = {0.0262698012, 0.0394047018, 0.0131349006, 
                                          0.0262698012, 0.0394047018, 0.0131349006, 
                                          0.1251019341, 0.110189418, 0.110189418, 
                                          0.1251019341, 0.110189418, 0.110189418};    
    
    std::vector<double> gradient = nn.getNumericGradient(inputData, inputLabels);
    assert(relativeError(gradient, expectedGradient) < 1e-6);
}

void numericGradientTest2()
{
    int inputLayerSize {2};
    std::vector<int> hiddenLayerSizes {2};
    int outputLayerSize {2};
    std::string lossFunction = "L2 norm";

    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, "leakyrelu", "linear", lossFunction); 
    
    nn.setWeights(std::vector<double> {0,-1, 1, 0, 0.5, 0.5, 0, 1, 1, 0, 0.5, -0.5});

    std::vector<double> inputData = {2, 1};
    std::vector<double> inputLabels = {2, 3};
    std::vector<double> expectedGradient = {-0.015835, -0.0316699, -0.015835, 
                                           0.357006, 0.714013, 0.357006,
                                          -0.138196, 0.0034549, -0.207294,
                                          -0.990405, 0.0247601, -1.4856074 };    
   
    std::vector<double> gradient = nn.getNumericGradient(inputData, inputLabels);
    assert(relativeError(gradient, expectedGradient) < 1e-6);
}

void backwardPropagationTest()
{

    int inputLayerSize {2};
    std::vector<int> hiddenLayerSizes {2};
    int outputLayerSize {2};
    std::string lossFunction = "None";

    //sigmoid was used in hidden layer just for the test
    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, "sigmoid", "sigmoid", lossFunction); 
    
    std::vector<double> inputData = {0.096, 0.5};
    std::vector<double> inputLabels = {6, 2};

    nn.forwardPropagation(inputData, inputLabels);    
    nn.backwardPropagation();
    std::vector<double> deltas = nn.getDeltas();
    std::vector<double> gradient = nn.getNumericGradient(inputData, inputLabels);

    assert(relativeError(deltas, gradient) < 1e-6);
}

void backwardPropagationTestL2()
{

    int inputLayerSize {2};
    std::vector<int> hiddenLayerSizes {2};
    int outputLayerSize {2};
    std::string lossFunction = "L2 norm";

    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, "leakyrelu", "linear", lossFunction); 
    
    nn.setWeights(std::vector<double> {0,-1, 1, 0, 0.5, 0.5, 0, 1, 1, 0, 0.5, -0.5});

    std::vector<double> inputData = {2.5, -1};
    std::vector<double> inputLabels = {2, 3};
   
    nn.forwardPropagation(inputData, inputLabels);    
    nn.backwardPropagation();
    std::vector<double> deltas = nn.getDeltas();
    std::vector<double> gradient = nn.getNumericGradient(inputData, inputLabels);

    assert(relativeError(deltas, gradient) < 1e-6);

}

void backwardPropagationTestSoftmax()
{

    int inputLayerSize {2};
    std::vector<int> hiddenLayerSizes {2};
    int outputLayerSize {2};
    std::string lossFunction = "Softmax";

    NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, "leakyrelu", "linear", lossFunction); 
    
    nn.setWeights(std::vector<double> {0,-1, 1, 0, 0.5, 0.5, 0, 1, 1, 0, 0.5, -0.5});

    std::vector<double> inputData = {2.5, -1};
    std::vector<double> inputLabels = {1};
   
    nn.forwardPropagation(inputData, inputLabels);    
    nn.backwardPropagation();
    std::vector<double> deltas = nn.getDeltas();
    std::vector<double> gradient = nn.getNumericGradient(inputData, inputLabels);

    assert(relativeError(deltas, gradient) < 1e-6);

    for(int i = 0; i<deltas.size(); i++){
        std::cout<<"From backprop : "<<deltas[i] <<"; From gradient :" << gradient[i]<<std::endl;
    }

}

void importDataSetTest(){
    DataSet dataSetTest("dataset_data_test.csv", "dataset_labels_test.csv");
    std::vector<std::vector<double>> inputData = dataSetTest.getInputData();
    std::vector<std::vector<double>> inputLabels = dataSetTest.getInputLabels();

    std::vector<std::vector<double>> expectedInputData = {{1,2.1,2},
                                                          {3,1,2},
                                                          {1,0,0},
                                                          {10.2,9,8}};


    std::vector<std::vector<double>> expectedInputLabels = {{1},
                                                            {2},
                                                            {0},
                                                            {1}};
    assert(inputData == expectedInputData);
    assert(inputLabels == expectedInputLabels);
}

int main()
{
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
