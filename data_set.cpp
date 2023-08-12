#include "data_set.hpp"
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>

DataSet::DataSet(std::string dataFilePath, std::string labelsFilePath){
   importCsv(dataFilePath, inputData);
   importCsv(labelsFilePath, inputLabels);
}

std::vector< std::vector <double>> DataSet::getInputData()
{
    return inputData;
}
std::vector< std::vector <double>> DataSet::getInputLabels()
{
    return inputLabels;
}

void DataSet::shuffle()
{
    unsigned int seed = unsigned(std::time(0));

    std::srand(seed);
    std::random_shuffle (inputData.begin(), inputData.end());
    std::srand(seed);
    std::random_shuffle(inputLabels.begin(), inputLabels.end());
}

void DataSet::importCsv(std::string filePath, std::vector< std::vector <double>> &matrix)
{
    std::ifstream in(filePath);
    std::string line;

    while (getline(in, line))                  
    {
        std::stringstream ss(line);                     
        std::vector<double> row;
        std::string data;
        while ( getline(ss, data, ',') )          
        {
            row.push_back(stod(data));            
        }
    if (row.size() > 0) matrix.push_back(row);    
   }
}