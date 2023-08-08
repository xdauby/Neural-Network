#ifndef EDGE_HPP
#define EDGE_HPP

#include <iostream>

class Node;

class Edge {

    private:

        double weight;
        double weightDelta;
        Node *incomingNode;
        Node *outgoingNode;

    public:
        Edge(Node *incomingNode, Node *outgoingNode);
        void setWeight(double value);
        void forwardPropagation(double value);
        double getWeight();
        double getWeightDelta();
        void backwardPropagation(double delta);

};


#endif