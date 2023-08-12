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
        void setPostActivationValue(double value);
        void setBias(double value);
        void setDelta(double value);
        void addDelta(double value);
        void reset();       
        double getPreActivationValue();
        double getPostActivationValue();
        std::vector<double> getIncomingWeights();
        std::vector<double> getIncomingDeltas();
        std::vector<Edge *> getIncomingEdges();
        void addToPreActivationValue(double value);
        void addIncomingEgde(Edge *edge);
        void addOutgoingEgde(Edge *edge);
        void forwardPropagation();
        void backwardPropagation();


};


#endif