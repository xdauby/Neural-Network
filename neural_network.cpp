#include "neural_network.hpp"

NeuralNetwork::NeuralNetwork(int inputLayerSize, std::vector<int> hiddenLayerSizes, int outputLayerSize) {
    
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
            
            layers[layerIndex].push_back(Node(nodeType, activationType));

            if(layerIndex > 0){
                //neural network fully connected
                for(unsigned int previousLayerNodeIndex {0}; previousLayerNodeIndex <  layers[layerIndex - 1].size(); previousLayerNodeIndex++){
                    Edge(&layers[layerIndex - 1][previousLayerNodeIndex], &layers[layerIndex][nodeIndex]);
                }
            }

        }

    }

}

double  NeuralNetwork::forwardPropagation(std::vector<double> inputData)
{
    if (inputData.size() == layers[0].size()) {

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





    } else {
       // throw error
    }


}

