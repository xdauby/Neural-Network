#ifndef NODE_HPP
#define NODE_HPP

#include <iostream>
#include <string>
#include <vector>
#include "node_type.hpp"
#include "activation_type.hpp"

typedef unsigned int uint;

class Edge;

class Node {
    /**
     * @brief Node of a Neural Network
     *
     */

   private:
    /**
     * @param preActivationValue is the computed pre activation value of this Node
     * @param postActivationValue is the computed post activation value of this Node
     * @param delta is the delta value of this node
     * @param bias is the bias value of this node
     * @param biasDelta is the bias delta value of this node
     * @param nodeType is the type of this node : INPUT, HIDDEN or OUTPUT
     * @param activationType is the activation function of this node : LEAKYRELU, LINEAR or SIGMOID
     * @param incomingEdges is the list of Edges going to this Node
     * @param outgoingEdges is the list of Edges leaving this Node
     *
     */
    double preActivationValue;
    double postActivationValue;
    double delta;
    double bias;
    double biasDelta;
    NodeType nodeType;
    ActivationType activationType;
    std::vector<Edge *> incomingEdges;
    std::vector<Edge *> outgoingEdges;

   public:
    /**
     * @brief construct a new Node object
     *
     * @param nodeType is the type of this node : INPUT, HIDDEN or OUTPUT
     * @param activationFunction is the activation function of this node : LEAKYRELU, LINEAR or SIGMOID
     */
    Node(NodeType nodeType, ActivationType activationType);

    /**
     * @brief set she pre activation value
     *
     * @param value
     */
    void setPreActivationValue(double value);

    /**
     * @brief set the sost activation value
     *
     * @param value
     */
    void setPostActivationValue(double value);

    /**
     * @brief set the bias value
     *
     * @param value
     */
    void setBias(double value);

    /**
     * @brief set the delta
     *
     * @param value
     */
    void setDelta(double value);

    /**
     * @brief add value to the current delta
     *
     * @param value
     */
    void addDelta(double value);

    /**
     * @brief reset Node values
     *
     */
    void reset();

    /**
     * @brief get the pre vctivation value
     *
     * @return the pre activation value of this Node
     */
    double getPreActivationValue();

    /**
     * @brief get the post activation value
     *
     * @return the post activation value of this Node
     */
    double getPostActivationValue();

    /**
     * @brief get the incoming weights
     *
     * @return vector of the incoming weights
     */
    std::vector<double> getIncomingWeights();

    /**
     * @brief get the incoming deltas
     *
     * @return vector of the incoming deltas
     */
    std::vector<double> getIncomingDeltas();

    /**
     * @brief get the incoming Edges object
     *
     * @return vector of incoming Edges
     */
    std::vector<Edge *> getIncomingEdges();

    /**
     * @brief add value to pre activation value of this Node
     *
     * @param value
     */
    void addToPreActivationValue(double value);

    /**
     * @brief add incoming Edge
     *
     * @param edge to add
     */
    void addIncomingEgde(Edge *edge);

    /**
     * @brief add outgoing Edge
     *
     * @param edge to add
     */
    void addOutgoingEgde(Edge *edge);

    /**
     * @brief perform random initialization
     *
     * @param bias is the new value of bias
     */
    void randomInitialization(double bias);

    /**
     * @brief perform forward propagation
     *
     */
    void forwardPropagation();

    /**
     * @brief perform backward propagation
     *
     */
    void backwardPropagation();
};

#endif