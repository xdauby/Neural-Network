#include "node.hpp"
#include <iostream>
#include <random>
#include "edge.hpp"

/**
 * @brief Construct a new Node:: Node object
 *
 * @param nodeType is the type of this node : "input", "hidden" or "output"
 * @param activationType is the activation type of thid node : "leakyrely", "sigmoid" or "linear"
 */
Node::Node(std::string nodeType, std::string activationType)
    : preActivationValue(0),
      postActivationValue(0),
      delta(0),
      bias(0),
      biasDelta(0),
      nodeType(nodeType),
      activationType(activationType) {}

/**
 * @brief set pre activation value
 *
 * @param value
 */
void Node::setPreActivationValue(double value) { preActivationValue = value; }

/**
 * @brief set post activation value
 *
 * @param value
 */
void Node::setPostActivationValue(double value) { postActivationValue = value; }

/**
 * @brief set bias
 *
 * @param value
 */
void Node::setBias(double value) { bias = value; }

/**
 * @brief set delta
 *
 * @param value
 */
void Node::setDelta(double value) { delta = value; }

/**
 * @brief add value to current delta
 *
 * @param value to add
 */
void Node::addDelta(double value) { delta += value; }

/**
 * @brief reset preActivationValue, postActivationValue, delta, biasDelta and WeightDelta of
 * incoming Edges.
 *
 */
void Node::reset() {
    preActivationValue = 0;
    postActivationValue = 0;
    delta = 0;
    biasDelta = 0;

    for (uint edgeIndex{0}; edgeIndex < incomingEdges.size(); edgeIndex++) {
        incomingEdges[edgeIndex]->setWeightDelta(0);
    }
}

/**
 * @brief get pre activation value
 *
 * @return pre activation value
 */
double Node::getPreActivationValue() { return preActivationValue; }

/**
 * @brief get post activation value
 *
 * @return post activation value
 */
double Node::getPostActivationValue() { return postActivationValue; }

/**
 * @brief get incoming edges
 *
 * @return vector of Edge
 */
std::vector<Edge *> Node::getIncomingEdges() { return incomingEdges; }

/**
 * @brief add value to pre activation value
 *
 * @param value
 */
void Node::addToPreActivationValue(double value) { preActivationValue += value; }

/**
 * @brief add edge to incomingEdges list
 *
 * @param edge
 */
void Node::addIncomingEgde(Edge *edge) { incomingEdges.push_back(edge); }

/**
 * @brief add edge to outgoingEdges list
 *
 * @param edge
 */
void Node::addOutgoingEgde(Edge *edge) { outgoingEdges.push_back(edge); }

/**
 * @brief perform forward propagation
 *
 */
void Node::forwardPropagation() {
    preActivationValue += bias;

    if (activationType == "linear") {
        postActivationValue = preActivationValue;
    } else if (activationType == "sigmoid") {
        postActivationValue = std::exp(preActivationValue) / (1 + std::exp(preActivationValue));
    } else if (activationType == "leakyrelu") {
        if (preActivationValue > 0) {
            postActivationValue = preActivationValue;
        } else {
            postActivationValue = 0.025 * preActivationValue;
        }
    }
    for (uint edgeIndex{0}; edgeIndex < outgoingEdges.size(); edgeIndex++) {
        outgoingEdges[edgeIndex]->forwardPropagation(postActivationValue);
    }
}

/**
 * @brief get incoming weights. First one the bias of the node, the following are the weights from
 * the incoming Edges.
 *
 * @return vector of incoming weights
 */
std::vector<double> Node::getIncomingWeights() {
    std::vector<double> incomingWeights;
    incomingWeights.push_back(bias);

    for (uint edgeIndex{0}; edgeIndex < incomingEdges.size(); edgeIndex++) {
        double edgeWeight = incomingEdges[edgeIndex]->getWeight();
        incomingWeights.push_back(edgeWeight);
    }
    return incomingWeights;
}

/**
 * @brief get incoming deltas. Same logic as above.
 *
 * @return vector of deltas
 */
std::vector<double> Node::getIncomingDeltas() {
    std::vector<double> incomingDeltas;
    incomingDeltas.push_back(biasDelta);

    for (uint edgeIndex{0}; edgeIndex < incomingEdges.size(); edgeIndex++) {
        double edgeDelta = incomingEdges[edgeIndex]->getWeightDelta();
        incomingDeltas.push_back(edgeDelta);
    }
    return incomingDeltas;
}

/**
 * @brief initialize randomly the weights of the incoming Edges.
 *
 * @param bias is the new value of the bias
 */
void Node::randomInitialization(double bias) {
    std::default_random_engine generator;
    std::normal_distribution<double> random_gaussian{0.0, 1.0};

    this->bias = bias;
    for (uint edgeIndex{0}; edgeIndex < incomingEdges.size(); edgeIndex++) {
        incomingEdges[edgeIndex]->setWeight(random_gaussian(generator) /
                                            std::sqrt(incomingEdges.size()));
    }
}

/**
 * @brief perform backward propagation
 *
 */
void Node::backwardPropagation() {
    if (activationType == "linear") {
        // delta *= 1; let's avoid useless computation
    } else if (activationType == "sigmoid") {
        delta *= postActivationValue * (1 - postActivationValue);
    } else if (activationType == "leakyrelu") {
        if (preActivationValue > 0) {
            // delta *= 1; let's avoid useless computation
        } else {
            delta *= 0.025;
        }
    }

    biasDelta = delta;

    for (uint edgeIndex{0}; edgeIndex < incomingEdges.size(); edgeIndex++) {
        incomingEdges[edgeIndex]->backwardPropagation(delta);
    }
}
