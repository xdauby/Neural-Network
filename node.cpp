#include "node.hpp"
#include "edge.hpp"
#include <cmath>

Node::Node(std::string nodeType, std::string activationType) : preActivationValue(0), 
                                                               postActivationValue(0), 
                                                               delta(0), 
                                                               bias(0), 
                                                               biasDelta(0), 
                                                               nodeType(nodeType), 
                                                               activationType(activationType)
{

}

        
void Node::setPreActivationValue(double value)
{
    preActivationValue = value;
}

double Node::getPreActivationValue()
{
    return preActivationValue;
}


void Node::setPostActivationValue(double value)
{
    postActivationValue = value;
}

double Node::getPostActivationValue()
{
    return postActivationValue;
}


void Node::addToPreActivationValue(double value)
{
    preActivationValue += value;
}


void Node::addIncomingEgde(Edge *edge)
{
    incomingEdges.push_back(edge); 
}

void Node::addOutgoingEgde(Edge *edge)
{
    outgoingEdges.push_back(edge); 
}


void Node::forwardPropagation()
{
   

    if (activationType == "linear") {
        postActivationValue = preActivationValue + bias;

    } else if (activationType == "sigmoid") {
        postActivationValue = std::exp(preActivationValue + bias) / (1 + std::exp(preActivationValue + bias));
    }    
    

    for(unsigned int edgeIndex {0}; edgeIndex < outgoingEdges.size(); edgeIndex++){

        outgoingEdges[edgeIndex]->forwardPropagation(postActivationValue);
    }
}
