#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

#include "data_set.hpp"
#include "edge.hpp"
#include "neural_network.hpp"
#include "node.hpp"
#include "loss_type.hpp"
#include "activation_type.hpp"

int main(int argc, char* argv[]) {

    try {
        std::string dataSetName = argv[1];
        double biases = std::stod(argv[2]);
        int epochs = std::stoi(argv[3]);
        double learningRate = std::stod(argv[4]);
        double mu = std::stod(argv[5]);
        std::vector<int> hiddenLayerSizes;
        for(int argi = 6; argi<argc; argi++){
            hiddenLayerSizes.push_back(std::stoi(argv[argi]));
        }

        std::string dataSetType;
        std::string inputDataPath;
        std::string inputLabelPath;

        ActivationType hiddenActivationType = ActivationType::LEAKYRELU;
        ActivationType outputActivationType;
        LossType lossType;
        int inputLayerSize;
        int outputLayerSize;

        if (dataSetName == "xo"){
            dataSetType = "classification";
            inputDataPath = "./datasets/classification_train_data.csv";
            inputLabelPath = "./datasets/classification_train_labels.csv";
            outputActivationType = ActivationType::SIGMOID;
            lossType = LossType::SOFTMAX;
            inputLayerSize = 100;
            outputLayerSize = 2;
        } else if (dataSetName == "y~x") {
            dataSetType = "regression";
            inputDataPath = "./datasets/regression_train_data.csv";
            inputLabelPath = "./datasets/regression_train_labels.csv";
            outputActivationType = ActivationType::LINEAR;
            lossType = LossType::L2NORM;
            inputLayerSize = 3;
            outputLayerSize = 1;
        } else {
            throw std::invalid_argument("Dataset name must be 'xo' or 'y~x'");
        }

        DataSet trainingSet(inputDataPath,
                            inputLabelPath, 
                            dataSetType);

        trainingSet.shuffle();
        NeuralNetwork nn(inputLayerSize, hiddenLayerSizes, outputLayerSize, hiddenActivationType, outputActivationType, lossType);
        nn.randomInitialization(biases);
        nn.train(trainingSet, epochs, learningRate, mu);
    } catch (const std::exception &e) {
        std::cout << "Args :" << std::endl;
        std::cout << "\t datasets : xo for classification or y~x for regression (string)" << std::endl;
        std::cout << "\t biases : values of bias (double) " << std::endl;
        std::cout << "\t epochs : number of epochs (int)" << std::endl;
        std::cout << "\t learning rate : learning rate coefficient (double) " << std::endl;
        std::cout << "\t mu : Nesterov coefficient (double)" << std::endl;
        std::cout << "\t hidden layer : number of nodes of each hidden layer (int ... int)" << std::endl;
        std::cout << "Execution examples :" << std::endl;
        std::cout << "\t ./main y~x 0 150 0.0001 0.95 2 2" << std::endl;
        std::cout << "\t ./main xo 0 50 0.003 0.95 8 8 6" << std::endl;
    }

}