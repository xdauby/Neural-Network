#ifndef EDGE_HPP
#define EDGE_HPP

#include <iostream>

class Node;
/**
 * @brief Edge of a Neural Network
 *
 */
class Edge {
   private:
    /**
     * @param weight is the weight of this Edge
     * @param weightDelta is the delta of this Edge
     * @param incomingNode is the starting Node of this Edge
     * @param outgoingNode is the ending Node of this Edge
     *
     */
    double weight;
    double weightDelta;
    Node *incomingNode;
    Node *outgoingNode;

   public:
    /**
     * @brief construct a new Edge object
     *
     * @param incomingNode
     * @param outgoingNode
     */
    Edge(Node *incomingNode, Node *outgoingNode);

    /**
     * @brief set the weight value
     *
     * @param value
     */
    void setWeight(double value);

    /**
     * @brief set the weight delta value
     *
     * @param value
     */
    void setWeightDelta(double value);

    /**
     * @brief perform forward propagation
     *
     * @param value
     */
    void forwardPropagation(double value);

    /**
     * @brief get the weight value
     *
     * @return the weight value
     */
    double getWeight();

    /**
     *
     * @brief get the weight delta
     *
     * @return the weight delta value
     */
    double getWeightDelta();

    /**
     * @brief perform backward propagation
     *
     * @param delta
     */
    void backwardPropagation(double delta);
};

#endif