#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <iostream>
#include <vector>
#include "data_set.hpp"
#include "edge.hpp"
#include "node.hpp"
#include "activation_type.hpp"
#include "loss_type.hpp"
#include "node_type.hpp"

typedef unsigned int uint;
typedef std::vector<std::vector<double>> matrix;

class NeuralNetwork {
    /**
     * @brief Neural Network
     *
     */
   private:
    /**
     * @param layers is a matrix of nodes
     * @param lossFunction is the loss : None, L2NORM or SOFTMAX
     * @param nWeights is the number of weights of this NN
     */
    std::vector<std::vector<Node>> layers;
    LossType lossFunction;
    uint nWeights;

   public:
    /**
     * @brief Construct a new Neural Network object
     *
     * @param inputLayerSize is the size of the first layer
     * @param hiddenLayerSizes is a vector that contains the size of each following layer
     * @param outputLayerSize is the size of the output layer
     * @param hiddenActivationType is the activation type of the Nodes of the hidden layer
     * @param outputActivationType is the Activation type of the last layer
     * @param lossFunction is the loss function : None, L2NORM or SOFTMAX
     */
    NeuralNetwork(int inputLayerSize, std::vector<int> hiddenLayerSizes, int outputLayerSize,
                  ActivationType hiddenActivationType, ActivationType outputActivationType,
                  LossType lossFunction);

    /**
     * @brief set weights object
     *
     * @param values is the vector of weights
     */
    void setWeights(std::vector<double> values);

    /**
     * @brief reset some values of this NN
     *
     */
    void reset();

    /**
     * @brief get number of weights
     *
     * @return the number of weights of this NN
     */
    uint getnWeights();

    /**
     * @brief get the Weights of this NN
     *
     * @return vector of weights
     */
    std::vector<double> getWeights();

    /**
     * @brief get the deltas of this NN
     *
     * @return vector of deltas
     */
    std::vector<double> getDeltas();

    /**
     * @brief get the post activation values of output Nodes
     *
     * @return vector of output values
     */
    std::vector<double> getOutputValues();

    /**
     * @brief random initialization of weights
     *
     * @param bias is the value given to all biases of this NN
     */
    void randomInitialization(double bias);

    /**
     * @brief perform forward propagation over this NN
     *
     * @param inputData is the vector of predictors of a sample
     * @param inputLabels is the labels of a sample
     * @return the computed loss
     */
    double forwardPropagation(std::vector<double> inputData, std::vector<double> inputLabels);

    /**
     * @brief perform backward propagation over this NN
     *
     */
    void backwardPropagation();

    /**
     * @brief get the numeric gradient of a sample
     *
     * @param inputData is the vector of predictors of a sample
     * @param inputLabels is the labels of a sample
     * @return the gradient
     */
    std::vector<double> getNumericGradient(std::vector<double> inputData,
                                           std::vector<double> inputLabels);

    /**
     * @brief train this NN. Nesterov momentum coupled to basic SGD is performed.
     *
     * @param dataset is the data set used to train the network
     * @param epochs is the number of epochs
     * @param learningRate is the learning rate
     * @param mu is the nesterov's parameter
     */
    void train(DataSet dataset, int epochs, double learningRate, double mu);

    /**
     * @brief compute accuracy of classifications
     *
     * @param dataset to pass to this NN
     * @return accuracy
     */
    double getAccuracy(DataSet dataset);
};

#endif