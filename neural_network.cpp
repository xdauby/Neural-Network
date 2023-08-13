#include "neural_network.hpp"
#include <stdexcept>
#include <cmath>

NeuralNetwork::NeuralNetwork(int inputLayerSize, 
                             std::vector<int> hiddenLayerSizes, 
                             int outputLayerSize,
                             std::string hiddenActivationType,
                             std::string outputActivationType,
                             std::string lossFunction) : lossFunction(lossFunction)
{
    nWeights = 0;
    layers.resize(2 + hiddenLayerSizes.size()); 
    std::string nodeType;
    std::string activationType;
    int layerSize;

    for(unsigned int layerIndex {0}; layerIndex < layers.size(); layerIndex++){
        
        if(layerIndex == 0){
            nodeType = "input";
            activationType = "linear"; 
            layerSize = inputLayerSize;
        } else if (layerIndex == (layers.size() - 1)) {
            nodeType = "output";
            activationType = outputActivationType;
            layerSize = outputLayerSize;
        } else {
            nodeType = "hidden";
            activationType = hiddenActivationType;
            layerSize = hiddenLayerSizes[layerIndex - 1];
        }
        

        for(int nodeIndex {0}; nodeIndex <  layerSize; nodeIndex++){
            Node *node = new Node(nodeType, activationType);
            layers[layerIndex].push_back(*node);
            
            //bias for layer 0 is set to 0 and not expected to change : we ignore 
            //those weights and start counting weights from layer 1
            if(layerIndex > 0){
                nWeights++;
            }
            
        }

    }

    //neural network fully connected
    for(unsigned int layerIndex {0}; layerIndex < layers.size() - 1; layerIndex++){
        for(unsigned int inputNodeIndex {0}; inputNodeIndex <  layers[layerIndex].size(); inputNodeIndex++){
            for(unsigned int outputNodeIndex {0}; outputNodeIndex <  layers[layerIndex + 1].size(); outputNodeIndex++){
                Node *input = &layers[layerIndex][inputNodeIndex];
                Node *output = &layers[layerIndex + 1][outputNodeIndex];
                new Edge(input, output);
                nWeights++;
                }
            }
        }

}

double  NeuralNetwork::forwardPropagation(std::vector<double> inputData, std::vector<double> inputLabels)
{

    double outputLayerIndex = layers.size() - 1;

    if (inputData.size() != layers[0].size()){
        throw std::invalid_argument("inputData size different from number of input Nodes !");
    }

    if (inputLabels.size() != layers[outputLayerIndex].size()){
        throw std::invalid_argument("inputLabels size different from number of output Nodes !");
    }

    //initialize input nodes
    for(unsigned int nodeIndex {0}; nodeIndex<layers[0].size(); nodeIndex++){
        layers[0][nodeIndex].setPreActivationValue(inputData[nodeIndex]);
    }

    //propagate forward
    for(unsigned int layerIndex {0}; layerIndex < layers.size(); layerIndex++){
        for(unsigned int nodeIndex {0}; nodeIndex <  layers[layerIndex].size(); nodeIndex++){            
            layers[layerIndex][nodeIndex].forwardPropagation();
        }
    }

    double computedLoss = 0;
    std::vector<double> outputValues = getOutputValues();

    if(lossFunction == "None"){
        for(unsigned int nodeIndex {0}; nodeIndex < layers[outputLayerIndex].size(); nodeIndex++){
            computedLoss += outputValues[nodeIndex];
            layers[outputLayerIndex][nodeIndex].setDelta(1.0);
        }
    } else if (lossFunction == "L2 norm") {
        for(unsigned int nodeIndex {0}; nodeIndex < layers[outputLayerIndex].size(); nodeIndex++){
            computedLoss += (outputValues[nodeIndex] - inputLabels[nodeIndex]) * (outputValues[nodeIndex] - inputLabels[nodeIndex]);
        }
        computedLoss = std::sqrt(computedLoss);

        for(unsigned int nodeIndex {0}; nodeIndex < layers[outputLayerIndex].size(); nodeIndex++){
            layers[outputLayerIndex][nodeIndex].setDelta( (outputValues[nodeIndex] - inputLabels[nodeIndex]) / computedLoss ); 
        }
    }

    return computedLoss;

}

void NeuralNetwork::setWeights(std::vector<double> values)
{
    if (values.size() != nWeights){
        throw std::invalid_argument("Vector of new weights (values) size different from number of weights !");
    }

    int cpt = 0;
    //bias for layer 0 is fixed to 0 and not expected to change : we ignore those weights and start from layer 1
    for(unsigned int nodeLayer {1}; nodeLayer < layers.size(); nodeLayer++){
        for(unsigned int nodeIndex {0}; nodeIndex < layers[nodeLayer].size(); nodeIndex++){
            double currentBias = values[cpt];
            layers[nodeLayer][nodeIndex].setBias(currentBias);
            cpt++;
            std::vector<Edge *> nodeIncomingEdges = layers[nodeLayer][nodeIndex].getIncomingEdges();
            for(unsigned int weightIndex {0}; weightIndex < nodeIncomingEdges.size(); weightIndex++){
                Edge *currentIncomingEdges = nodeIncomingEdges[weightIndex];
                currentIncomingEdges->setWeight(values[cpt]);
                cpt++;
            }
        }
    }
}

void NeuralNetwork::reset()
{
    for(unsigned int layerIndex {0}; layerIndex < layers.size(); layerIndex++){
        for(unsigned int nodeIndex {0}; nodeIndex <  layers[layerIndex].size(); nodeIndex++){            
            layers[layerIndex][nodeIndex].reset();
        }
    }
}

std::vector<double> NeuralNetwork::getOutputValues()
{
    std::vector<double> outputValues;   
    double outputLayerIndex = layers.size() - 1;
    for(unsigned int nodeIndex {0}; nodeIndex < layers[outputLayerIndex].size(); nodeIndex++){
        outputValues.push_back(layers[outputLayerIndex][nodeIndex].getPostActivationValue());
    }
    return outputValues;
}


std::vector<double> NeuralNetwork::getWeights()
{
    std::vector<double> weights;
    //bias for layer 0 is fixed to 0 and not expected to change : we ignore those weights and start from layer 1
    for(unsigned int nodeLayer {1}; nodeLayer < layers.size(); nodeLayer++){
        for(unsigned int nodeIndex {0}; nodeIndex < layers[nodeLayer].size(); nodeIndex++){
            std::vector<double> nodeWeights = layers[nodeLayer][nodeIndex].getIncomingWeights();
            for(unsigned int weightIndex {0}; weightIndex < nodeWeights.size(); weightIndex++){
                weights.push_back(nodeWeights[weightIndex]);
            }
        }
    }
    return weights;
}

std::vector<double> NeuralNetwork::getDeltas()
{
    std::vector<double> deltas;
    //bias for layer 0 is set to 0 : we ignore those weights and start from layer 1, therefore we do not
    //update biasDelta from layer 0
    for(unsigned int nodeLayer {1}; nodeLayer < layers.size(); nodeLayer++){
        for(unsigned int nodeIndex {0}; nodeIndex < layers[nodeLayer].size(); nodeIndex++){
            std::vector<double> nodeDeltas = layers[nodeLayer][nodeIndex].getIncomingDeltas();
            for(unsigned int deltaIndex {0}; deltaIndex < nodeDeltas.size(); deltaIndex++){
                deltas.push_back(nodeDeltas[deltaIndex]);
            }
        }
    }
    return deltas;
}

unsigned int NeuralNetwork::getnWeights()
{
    return nWeights;
}

void NeuralNetwork::backwardPropagation()
{
    for(int nodeLayer = layers.size() - 1; nodeLayer >= 1; nodeLayer--){
        for(unsigned int nodeIndex {0}; nodeIndex < layers[nodeLayer].size(); nodeIndex++){
            layers[nodeLayer][nodeIndex].backwardPropagation();
        }
    }
}

void NeuralNetwork::train(DataSet dataSet, int epochs, double learningRate)
{
    unsigned int nRows = dataSet.getnRows();
    
    for(int epochNumber = 0; epochNumber < epochs; epochNumber++){
        
        dataSet.shuffle();
        std::vector<double> weights = getWeights();
        std::vector<std::vector<double>> inputData = dataSet.getInputData();
        std::vector<std::vector<double>> inputLabels = dataSet.getInputLabels();  

        for(unsigned int rowNumber = 0; rowNumber < nRows; rowNumber++){
            forwardPropagation(inputData[rowNumber], inputLabels[rowNumber]);
            backwardPropagation();
            std::vector<double> gradient = getDeltas();
            for(unsigned int weightNumber = 0; weightNumber < weights.size(); weightNumber++){
                weights[weightNumber] -= learningRate*gradient[weightNumber];
            }
            setWeights(weights);
        }

        double sumLoss = 0;
        for(unsigned int rowNumber = 0; rowNumber < nRows; rowNumber++){
            sumLoss += forwardPropagation(inputData[rowNumber], inputLabels[rowNumber]);
        }

    }
}