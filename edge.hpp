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
};


#endif