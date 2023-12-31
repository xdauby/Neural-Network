#include "neural_network.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

/**
 * @brief Construct a new Neural Network:: Neural Network.
 * All layers are fully connected to the previous one.
 *
 * @param inputLayerSize is the size of the first layer
 * @param hiddenLayerSizes is a vector that contains the size of each following layer
 * @param outputLayerSize is the size of the output layer
 * @param hiddenActivationType is the activation type of the Nodes of the hidden layer
 * @param outputActivationType is the Activation type of the last layer
 * @param lossFunction is the loss function : NONE , L2NORM or SOFTMAX
 */
NeuralNetwork::NeuralNetwork(int inputLayerSize, std::vector<int> hiddenLayerSizes,
                             int outputLayerSize, ActivationType hiddenActivationType,
                             ActivationType outputActivationType, LossType lossFunction)
    : lossFunction(lossFunction) {
    nWeights = 0;
    layers.resize(2 + hiddenLayerSizes.size());
    NodeType nodeType;
    ActivationType activationType;
    int layerSize;

    for (uint layerIndex{0}; layerIndex < layers.size(); layerIndex++) {
        if (layerIndex == 0) {
            nodeType = NodeType::INPUT;
            activationType = ActivationType::LINEAR;
            layerSize = inputLayerSize;
        } else if (layerIndex == (layers.size() - 1)) {
            nodeType = NodeType::OUTPUT;
            activationType = outputActivationType;
            layerSize = outputLayerSize;
        } else {
            nodeType = NodeType::HIDDEN;
            activationType = hiddenActivationType;
            layerSize = hiddenLayerSizes[layerIndex - 1];
        }

        for (int nodeIndex{0}; nodeIndex < layerSize; nodeIndex++) {
            Node *node = new Node(nodeType, activationType);
            layers[layerIndex].push_back(*node);

            // bias for layer 0 is set to 0 and not expected to change : we ignore
            // those weights and start counting weights from layer 1
            if (layerIndex > 0) {
                nWeights++;
            }
        }
    }

    // neural network fully connected
    for (uint layerIndex{0}; layerIndex < layers.size() - 1; layerIndex++) {
        for (uint inputNodeIndex{0}; inputNodeIndex < layers[layerIndex].size(); inputNodeIndex++) {
            for (uint outputNodeIndex{0}; outputNodeIndex < layers[layerIndex + 1].size();
                 outputNodeIndex++) {
                new Edge(&layers[layerIndex][inputNodeIndex],
                         &layers[layerIndex + 1][outputNodeIndex]);
                nWeights++;
            }
        }
    }
}

/**
 * @brief perform forward propagation
 *
 * @param inputData is the vector of predictors of a sample
 * @param inputLabels is the labels of a sample
 * @return the computed loss
 */
double NeuralNetwork::forwardPropagation(std::vector<double> inputData,
                                         std::vector<double> inputLabels) {
    reset();
    int outputLayerIndex = layers.size() - 1;

    if (inputData.size() != layers[0].size()) {
        throw std::invalid_argument("inputData size different from number of input Nodes !");
    }

    // if (inputLabels.size() != layers[outputLayerIndex].size()){
    //     throw std::invalid_argument("inputLabels size different from number of output Nodes !");
    // }

    // initialize input nodes
    for (uint nodeIndex{0}; nodeIndex < layers[0].size(); nodeIndex++) {
        layers[0][nodeIndex].setPreActivationValue(inputData[nodeIndex]);
    }

    // propagate forward
    for (uint layerIndex{0}; layerIndex < layers.size(); layerIndex++) {
        for (uint nodeIndex{0}; nodeIndex < layers[layerIndex].size(); nodeIndex++) {
            layers[layerIndex][nodeIndex].forwardPropagation();
        }
    }

    double computedLoss = 0.0;
    std::vector<double> outputValues = getOutputValues();

    if (lossFunction == LossType::NONE) {
        for (uint nodeIndex{0}; nodeIndex < layers[outputLayerIndex].size(); nodeIndex++) {
            computedLoss += outputValues[nodeIndex];
            layers[outputLayerIndex][nodeIndex].setDelta(1.0);
        }
    } else if (lossFunction == LossType::L2NORM) {
        for (uint nodeIndex{0}; nodeIndex < layers[outputLayerIndex].size(); nodeIndex++) {
            computedLoss += (outputValues[nodeIndex] - inputLabels[nodeIndex]) *
                            (outputValues[nodeIndex] - inputLabels[nodeIndex]);
        }
        computedLoss = std::sqrt(computedLoss);

        for (uint nodeIndex{0}; nodeIndex < layers[outputLayerIndex].size(); nodeIndex++) {
            layers[outputLayerIndex][nodeIndex].setDelta(
                (outputValues[nodeIndex] - inputLabels[nodeIndex]) / computedLoss);
        }
    } else if (lossFunction == LossType::SOFTMAX) {
        double sumExp = 0.0;
        uint label = int(inputLabels[0]);

        for (uint nodeIndex{0}; nodeIndex < layers[outputLayerIndex].size(); nodeIndex++) {
            sumExp += std::exp(outputValues[nodeIndex]);
        }

        for (uint nodeIndex{0}; nodeIndex < layers[outputLayerIndex].size(); nodeIndex++) {
            if (label != nodeIndex) {
                layers[outputLayerIndex][nodeIndex].setDelta(std::exp(outputValues[nodeIndex]) /
                                                             sumExp);
            } else {
                layers[outputLayerIndex][nodeIndex].setDelta(
                    (std::exp(outputValues[nodeIndex]) / sumExp) - 1);
                computedLoss = -std::log(std::exp(outputValues[nodeIndex]) / sumExp);
            }
        }
    }

    return computedLoss;
}

/**
 * @brief set weights
 *
 * @param values is a vector of weights
 */
void NeuralNetwork::setWeights(std::vector<double> values) {
    if (values.size() != nWeights) {
        throw std::invalid_argument(
            "Vector of new weights (values) size different from number of weights !");
    }

    int cpt = 0;
    // bias for layer 0 is fixed to 0 and not expected to change : we ignore those weights and start
    // from layer 1
    for (uint nodeLayer{1}; nodeLayer < layers.size(); nodeLayer++) {
        for (uint nodeIndex{0}; nodeIndex < layers[nodeLayer].size(); nodeIndex++) {
            double currentBias = values[cpt];
            layers[nodeLayer][nodeIndex].setBias(currentBias);
            cpt++;
            std::vector<Edge *> nodeIncomingEdges = layers[nodeLayer][nodeIndex].getIncomingEdges();
            for (uint weightIndex{0}; weightIndex < nodeIncomingEdges.size(); weightIndex++) {
                Edge *currentIncomingEdges = nodeIncomingEdges[weightIndex];
                currentIncomingEdges->setWeight(values[cpt]);
                cpt++;
            }
        }
    }
}

/**
 * @brief reset values from Node : check reset comments of Node
 *
 */
void NeuralNetwork::reset() {
    for (uint layerIndex{0}; layerIndex < layers.size(); layerIndex++) {
        for (uint nodeIndex{0}; nodeIndex < layers[layerIndex].size(); nodeIndex++) {
            layers[layerIndex][nodeIndex].reset();
        }
    }
}

/**
 * @brief get the post activation values of output Nodes
 *
 * @return vector of output values
 */
std::vector<double> NeuralNetwork::getOutputValues() {
    std::vector<double> outputValues;
    double outputLayerIndex = layers.size() - 1;
    for (uint nodeIndex{0}; nodeIndex < layers[outputLayerIndex].size(); nodeIndex++) {
        outputValues.push_back(layers[outputLayerIndex][nodeIndex].getPostActivationValue());
    }
    return outputValues;
}

/**
 * @brief get weights of this NN
 *
 * @return vector of weights
 */
std::vector<double> NeuralNetwork::getWeights() {
    std::vector<double> weights;
    // bias for layer 0 is fixed to 0 and not expected to change : we ignore those weights and start
    // from layer 1
    for (uint nodeLayer{1}; nodeLayer < layers.size(); nodeLayer++) {
        for (uint nodeIndex{0}; nodeIndex < layers[nodeLayer].size(); nodeIndex++) {
            std::vector<double> nodeWeights = layers[nodeLayer][nodeIndex].getIncomingWeights();
            for (uint weightIndex{0}; weightIndex < nodeWeights.size(); weightIndex++) {
                weights.push_back(nodeWeights[weightIndex]);
            }
        }
    }
    return weights;
}

/**
 * @brief get deltas of this NN
 *
 * @return vector of deltas
 */
std::vector<double> NeuralNetwork::getDeltas() {
    std::vector<double> deltas;
    // bias for layer 0 is set to 0 : we ignore those weights and start from layer 1, therefore we
    // do not update biasDelta from layer 0
    for (uint nodeLayer{1}; nodeLayer < layers.size(); nodeLayer++) {
        for (uint nodeIndex{0}; nodeIndex < layers[nodeLayer].size(); nodeIndex++) {
            std::vector<double> nodeDeltas = layers[nodeLayer][nodeIndex].getIncomingDeltas();
            for (uint deltaIndex{0}; deltaIndex < nodeDeltas.size(); deltaIndex++) {
                deltas.push_back(nodeDeltas[deltaIndex]);
            }
        }
    }
    return deltas;
}

/**
 * @brief get the number of weights of this NN
 *
 * @return number of weights
 */
uint NeuralNetwork::getnWeights() { return nWeights; }

void NeuralNetwork::randomInitialization(double bias) {
    for (uint nodeLayer{1}; nodeLayer < layers.size(); nodeLayer++) {
        for (uint nodeIndex{0}; nodeIndex < layers[nodeLayer].size(); nodeIndex++) {
            layers[nodeLayer][nodeIndex].randomInitialization(bias);
        }
    }
}

/**
 * @brief perform backward propagation over this NN
 *
 */
void NeuralNetwork::backwardPropagation() {
    for (int nodeLayer = layers.size() - 1; nodeLayer >= 1; nodeLayer--) {
        for (uint nodeIndex{0}; nodeIndex < layers[nodeLayer].size(); nodeIndex++) {
            layers[nodeLayer][nodeIndex].backwardPropagation();
        }
    }
}

/**
 * @brief compute numeric gradient. Helps for tests.
 *
 * @param inputData is the vector of predictors of a sample
 * @param inputLabels is the labels of a sample
 * @return std::vector<double>
 */
std::vector<double> NeuralNetwork::getNumericGradient(std::vector<double> inputData,
                                                      std::vector<double> inputLabels) {
    std::vector<double> gradient;
    std::vector<double> weights = getWeights();
    std::vector<double> weightsCopy = weights;

    double lossLeft;
    double lossRight;
    double eps = 0.000000001;

    for (uint weightNumber = 0; weightNumber < weights.size(); weightNumber++) {
        weightsCopy[weightNumber] = weights[weightNumber] - eps;
        setWeights(weightsCopy);
        lossLeft = forwardPropagation(inputData, inputLabels);

        weightsCopy[weightNumber] = weights[weightNumber] + eps;
        setWeights(weightsCopy);
        lossRight = forwardPropagation(inputData, inputLabels);

        gradient.push_back((lossRight - lossLeft) / (2.0 * eps));

        weightsCopy[weightNumber] = weights[weightNumber];
    }
    setWeights(weights);
    return gradient;
}

/**
 * @brief train this NN. Nesterov momentum coupled to basic SGD is performed.
 *
 * @param dataset is the data set used to train the network
 * @param epochs is the number of epochs
 * @param learningRate is the learning rate
 * @param mu is the nesterov's parameter
 */
void NeuralNetwork::train(DataSet dataset, int epochs, double learningRate, double mu) {
    uint nRows = dataset.getnRows();
    std::vector<double> velocity(nWeights, 0.0);

    for (int epochNumber = 0; epochNumber < epochs; epochNumber++) {
        dataset.shuffle();

        matrix inputData = dataset.getInputData();
        matrix inputLabels = dataset.getInputLabels();

        for (uint rowNumber = 1; rowNumber < nRows; rowNumber++) {
            std::vector<double> previousVelocity = velocity;

            forwardPropagation(inputData[rowNumber], inputLabels[rowNumber]);
            backwardPropagation();
            std::vector<double> weights = getWeights();
            std::vector<double> gradient = getDeltas();
            for (uint weightNumber = 0; weightNumber < weights.size(); weightNumber++) {
                velocity[weightNumber] =
                    mu * velocity[weightNumber] - learningRate * gradient[weightNumber];
                weights[weightNumber] +=
                    (-mu * previousVelocity[weightNumber]) + ((1 + mu) * velocity[weightNumber]);
            }
            setWeights(weights);
        }

        double error = 0;
        for (uint rowNumber = 0; rowNumber < nRows; rowNumber++) {
            error += forwardPropagation(inputData[rowNumber], inputLabels[rowNumber]);
        }

        if (dataset.getType() == "classification") {
            std::cout << "Epoch : " << epochNumber << ", Loss over training set :" << error / nRows
                      << ", Accuracy over training set : " << getAccuracy(dataset) <<'%'<< std::endl;
        } else {
            std::cout << "Epoch : " << epochNumber << ", Loss :" << error / nRows << std::endl;
        }
    }
}

/**
 * @brief compute accuracy of classifications
 *
 * @param dataset to pass to this NN
 * @return accuracy in %
 */
double NeuralNetwork::getAccuracy(DataSet dataset) {
    matrix inputData = dataset.getInputData();
    matrix inputLabels = dataset.getInputLabels();
    int nRows = dataset.getnRows();

    int trueLabels = 0;

    for (int currentRow = 0; currentRow < nRows; currentRow++) {
        forwardPropagation(inputData[currentRow], inputLabels[currentRow]);
        std::vector<double> outputValues = getOutputValues();
        std::vector<double>::iterator max =
            std::max_element(outputValues.begin(), outputValues.end());
        int predictedLabel = std::distance(outputValues.begin(), max);

        if (predictedLabel == int(inputLabels[currentRow][0])) {
            trueLabels += 1;
        }
    }

    return (double)trueLabels * 100 / nRows;
}