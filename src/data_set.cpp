#include "data_set.hpp"
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>

/**
 * @brief Construct a new Data Set:: Data Set
 * 
 * @param dataFilePath is the path of the predictors 
 * @param labelsFilePath is the path of the labels
 * @param datasetType is the type of dataset : "classification" or "regression"
 */
DataSet::DataSet(std::string dataFilePath, std::string labelsFilePath, std::string datasetType): datasetType(datasetType)
{
   importCsv(dataFilePath, inputData);
   importCsv(labelsFilePath, inputLabels);
}

/**
 * @brief get predictors
 * 
 * @return matrix of predictors 
 */
matrix DataSet::getInputData()
{
    return inputData;
}

/**
 * @brief get labels
 * 
 * @return matrix of labels 
 */
matrix DataSet::getInputLabels()
{
    return inputLabels;
}

/**
 * @brief get number of rows
 * 
 * @return numbe of rows
 */
unsigned int DataSet::getnRows()
{
    return inputData.size();
}

/**
 * @brief get type of dataset
 * 
 * @return type name 
 */
std::string DataSet::getType()
{
    return datasetType;
}

/**
 * @brief shuffle the data
 * 
 */
void DataSet::shuffle()
{
    unsigned int seed = unsigned(std::time(0));

    std::srand(seed);
    std::random_shuffle (inputData.begin(), inputData.end());
    std::srand(seed);
    std::random_shuffle(inputLabels.begin(), inputLabels.end());
}

/**
 * @brief import a csv file
 * 
 * @param filePath is the file path
 * @param matrix is the matrix to fill in the values
 */
void DataSet::importCsv(std::string filePath, matrix &matrix)
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