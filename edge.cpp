#include "edge.hpp"
#include "node.hpp"

/**
 * @brief construct a new Edge:: Edge object
 *
 * @param incomingNode is the starting Node of this Edge
 * @param outgoingNode is the ending Node of this Edge
 */
Edge::Edge(Node *incomingNode, Node *outgoingNode)
    : weight(1), weightDelta(0), incomingNode(incomingNode), outgoingNode(outgoingNode) {
    outgoingNode->addIncomingEgde(this);
    incomingNode->addOutgoingEgde(this);
}

/**
 * @brief set weight value
 *
 * @param value
 */
void Edge::setWeight(double value) { weight = value; }

/**
 * @brief set delta value
 *
 * @param value
 */
void Edge::setWeightDelta(double value) { weightDelta = value; }

/**
 * @brief perform forward propagation
 *
 * @param value propagated from incoming Node
 */
void Edge::forwardPropagation(double value) {
    outgoingNode->addToPreActivationValue(value * weight);
}

/**
 * @brief get weight value
 *
 * @return weight value
 */
double Edge::getWeight() { return weight; }

/**
 * @brief get weight delta
 *
 * @return weight delta
 */
double Edge::getWeightDelta() { return weightDelta; }

/**
 * @brief perform backward propagation
 *
 * @param delta backpropagated from outgoing Node
 */
void Edge::backwardPropagation(double delta) {
    weightDelta = delta * incomingNode->getPostActivationValue();
    incomingNode->addDelta(delta * weight);
}
