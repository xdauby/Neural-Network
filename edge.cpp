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
void Edge::setWeight(double value)
{
    weight = value;
}

void Edge::forwardPropagation(double value)
{
    outgoingNode->addToPreActivationValue(value * weight);
}

double Edge::getWeight()
{
    return weight;
}

double Edge::getWeightDelta()
{
    return weightDelta;
}

void Edge::backwardPropagation(double delta)
{
    weightDelta =  delta * incomingNode->getPostActivationValue();
    incomingNode->addDelta(delta * weight);
}
