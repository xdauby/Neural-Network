#ifndef NODE_HPP
#define NODE_HPP

#include <iostream>
#include <vector>
#include <string> 

class Edge;

class Node{

    private:

        
        

    public:
        
        double preActivationValue;
        double postActivationValue;
        double delta;
        double bias;
        double biasDelta;

        std::string nodeType;
        std::string activationType;

        std::vector<Edge *> incomingEdges;
        std::vector<Edge *> outgoingEdges;


        Node(std::string nodeType, std::string activationType);
        
        void setPreActivationValue(double value);
        double getPreActivationValue();
        void setPostActivationValue(double value);
        double getPostActivationValue();
        std::vector<double> getIncomingWeights();
        void addToPreActivationValue(double value);
        void addIncomingEgde(Edge *edge);
        void addOutgoingEgde(Edge *edge);
        void forwardPropagation();
        
        void test(){};


};


#endif