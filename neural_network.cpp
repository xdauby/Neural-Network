#include "neural_network.hpp"

NeuralNetwork::NeuralNetwork(int inputLayerSize, std::vector<int> hiddenLayerSizes, int outputLayerSize) {
    
    layers.resize(2 + hiddenLayerSizes.size()); 

    std::string nodeType;
    std::string activationType;
    int layerSize;

    for(unsigned int layerNumber {0}; layerNumber < layers.size(); layerNumber++){
        
        if(layerNumber == 0){
            nodeType = "input";
            activationType = "linear";
            layerSize = inputLayerSize;
        } else if (layerNumber == (layers.size() - 1)) {
            nodeType = "output";
            activationType = "sigmoid";
            layerSize = outputLayerSize;
        } else {
            nodeType = "hidden";
            activationType = "sigmoid";
            layerSize = hiddenLayerSizes[layerNumber];
        }

        for(int NodeNumber {0}; NodeNumber <  layerSize; NodeNumber++){
            layers[layerNumber][NodeNumber] = Node(nodeType, activationType);
        }

    }

}