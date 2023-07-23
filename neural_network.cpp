#include "neural_network.hpp"
#include <stdexcept>

NeuralNetwork::NeuralNetwork(int inputLayerSize, 
                             std::vector<int> hiddenLayerSizes, 
                             int outputLayerSize, 
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
            activationType = "sigmoid";
            layerSize = outputLayerSize;
        } else {
            nodeType = "hidden";
            activationType = "sigmoid";
            layerSize = hiddenLayerSizes[layerIndex - 1];
        }

        for(int nodeIndex {0}; nodeIndex <  layerSize; nodeIndex++){
            Node *node = new Node(nodeType, activationType);
            layers[layerIndex].push_back(*node);
            nWeights++;
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

    double computedLoss = 0;
    double outputLayerIndex = layers.size() - 1;
    std::vector<double> outputValues = getOutputValues();

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
    for(unsigned int layerIndex {0}; layerIndex < layers.size() - 1; layerIndex++){
        for(unsigned int nodeIndex {0}; nodeIndex <  layers[layerIndex].size(); nodeIndex++){
            std::cout << "Layer : " << layerIndex << ", Node : "<< nodeIndex <<std::endl;
            std::cout << "Pre Activation Value : " << layers[layerIndex][nodeIndex].getPreActivationValue() <<std::endl;

            layers[layerIndex][nodeIndex].forwardPropagation();
                            
            std::cout << "Post Activation Value : " << layers[layerIndex][nodeIndex].getPostActivationValue() <<std::endl;
            std::cout<<std::endl;
        }
    }

    

    if(lossFunction == "None"){
        for(unsigned int nodeIndex {0}; nodeIndex < layers[outputLayerIndex].size(); nodeIndex++){
            computedLoss += outputValues[nodeIndex];
        }
    } else if (lossFunction == "L2 norm") {
        for(unsigned int nodeIndex {0}; nodeIndex < layers[outputLayerIndex].size(); nodeIndex++){
            computedLoss += (outputValues[nodeIndex] - inputLabels[nodeIndex]) * (outputValues[nodeIndex] - inputLabels[nodeIndex]);
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
    for(unsigned int nodeLayer {0}; nodeLayer < layers.size(); nodeLayer++){
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
    for(unsigned int nodeLayer {0}; nodeLayer < layers.size(); nodeLayer++){
        for(unsigned int nodeIndex {0}; nodeIndex < layers[nodeLayer].size(); nodeIndex++){
            std::vector<double> nodeWeights = layers[nodeLayer][nodeIndex].getIncomingWeights();
            for(unsigned int weightIndex {0}; weightIndex < nodeWeights.size(); weightIndex++){
                weights.push_back(nodeWeights[weightIndex]);
            }
        }
    }
    return weights;
}