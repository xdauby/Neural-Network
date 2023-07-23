#include "edge.hpp"
#include "node.hpp"

Edge::Edge(Node *incomingNode, Node *outgoingNode) : weight(1), 
                                                     weightDelta(0), 
                                                     incomingNode(incomingNode), 
                                                     outgoingNode(outgoingNode) 
{
    outgoingNode->addIncomingEgde(this);
    incomingNode->addOutgoingEgde(this);

}



void Edge::forwardPropagation(double value)
{
    outgoingNode->addToPreActivationValue(value * weight);
}

double Edge::getWeight()
{
    return weight;
}