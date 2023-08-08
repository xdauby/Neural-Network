#include "node.hpp"
#include "edge.hpp"
#include <cmath>

Node::Node(std::string nodeType, std::string activationType) : preActivationValue(0), 
                                                               postActivationValue(0), 
                                                               delta(1), 
                                                               bias(0), 
                                                               biasDelta(1), 
                                                               nodeType(nodeType), 
                                                               activationType(activationType)
{

}

        
void Node::setPreActivationValue(double value)
{
    preActivationValue = value;
}

void Node::setPostActivationValue(double value)
{
    postActivationValue = value;
}

void Node::setBias(double value)
{
    bias = value;
}

void Node::setDelta(double value)
{
    delta = value;
}       


double Node::getPreActivationValue()
{
    return preActivationValue;
}

double Node::getPostActivationValue()
{
    return postActivationValue;
}

std::vector<Edge *> Node::getIncomingEdges()
{
    return incomingEdges;
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
    preActivationValue += bias;
    if (activationType == "linear") {
        postActivationValue = preActivationValue;

    } else if (activationType == "sigmoid") {
        postActivationValue = std::exp(preActivationValue) / (1 + std::exp(preActivationValue));
    }    
    
    for(unsigned int edgeIndex {0}; edgeIndex < outgoingEdges.size(); edgeIndex++){
        outgoingEdges[edgeIndex]->forwardPropagation(postActivationValue);
    }
}


std::vector<double> Node::getIncomingWeights()
{
    std::vector<double> incomingWeights;
    incomingWeights.push_back(bias);
    for(unsigned int edgeIndex {0}; edgeIndex < incomingEdges.size(); edgeIndex++){
        double edgeWeight = incomingEdges[edgeIndex]->getWeight();
        incomingWeights.push_back(edgeWeight);
    }
    return incomingWeights;
}

std::vector<double> Node::getIncomingDeltas()
{
    std::vector<double> incomingDeltas;
    incomingDeltas.push_back(biasDelta);
    for(unsigned int edgeIndex {0}; edgeIndex < incomingEdges.size(); edgeIndex++){
        double edgeDelta = incomingEdges[edgeIndex]->getWeightDelta();
        incomingDeltas.push_back(edgeDelta);
    }
    return incomingDeltas;
}


void Node::backwardPropagation()
{
    if (activationType == "linear") {
        delta *= 1;
    } else if (activationType == "sigmoid") {
        delta *= postActivationValue * (1 - postActivationValue);
    }

    biasDelta = delta;

    for(unsigned int edgeIndex {0}; edgeIndex < incomingEdges.size(); edgeIndex++){
        incomingEdges[edgeIndex]->backwardPropagation(delta);//
    }

}
