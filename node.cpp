#include "node.hpp"

Node::Node(std::string nodeType, std::string activationType) : preActivationValue(0), 
                                                               postActivationValue(0), 
                                                               delta(0), 
                                                               bias(0), 
                                                               biasDelta(0), 
                                                               nodeType(nodeType), 
                                                               activationType(activationType)
{

}

void Node::addIncomingEgde(Edge edge)
{
    incomingEdges.push_back(edge); 
}

void Node::addOutgoingEgde(Edge edge)
{
    outgoingEdges.push_back(edge); 
}