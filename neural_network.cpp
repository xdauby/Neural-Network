#include "neural_network.hpp"

NeuralNetwork::NeuralNetwork(int inputLayerSize, 
                             std::vector<int> hiddenLayerSizes, 
                             int outputLayerSize, 
                             std::string lossFunction) : lossFunction(lossFunction)
{
    
    this->layers.resize(2 + hiddenLayerSizes.size()); 

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
            this->layers[layerIndex].push_back(*node);

        }

    }

    //neural network fully connected
    for(unsigned int layerIndex {0}; layerIndex < layers.size() - 1; layerIndex++){
            for(unsigned int inputNodeIndex {0}; inputNodeIndex <  layers[layerIndex].size(); inputNodeIndex++){
                for(unsigned int outputNodeIndex {0}; outputNodeIndex <  layers[layerIndex + 1].size(); outputNodeIndex++){
                    
                    Node *input = &layers[layerIndex][inputNodeIndex];
                    Node *output = &layers[layerIndex + 1][outputNodeIndex];
                    
                    new Edge(input, output);
                
                }
            }
        }
    


    //layers[0][0].forwardPropagation();
  


}

double  NeuralNetwork::forwardPropagation(std::vector<double> inputData)
{

    if (inputData.size() == layers[0].size()) {

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

        double computedLoss = 0;

        if(lossFunction == "None"){
            double outputLayerSize = layers.size() - 1;
            for(unsigned int nodeIndex {0}; nodeIndex < layers[outputLayerSize].size(); nodeIndex++){
                computedLoss += layers[outputLayerSize][nodeIndex].getPostActivationValue();
            }
        }

        return computedLoss;





    } else {

    }


}

