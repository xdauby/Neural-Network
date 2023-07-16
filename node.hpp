#ifndef NODE_HPP
#define NODE_HPP

#include <iostream>
#include <vector>
#include <string> 

class Edge;

class Node{

    private:

        double preActivationValue;
        double postActivationValue;
        double delta;
        double bias;
        double biasDelta;

        std::string nodeType;
        std::string activationType;

        std::vector<Edge *> incomingEdges;
        std::vector<Edge *> outgoingEdges;

    public:

        Node(std::string nodeType, std::string activationType);
        
        void setPreActivationValue(double value);
        double getPreActivationValue();
        
        void addToPreActivationValue(double value);

        void addIncomingEgde(Edge *edge);
        void addOutgoingEgde(Edge *edge);
        void forwardPropagation();


};


#endif