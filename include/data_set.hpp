#ifndef DATA_SET_HPP
#define DATA_SET_HPP

#include <vector>
#include <string>

typedef std::vector<std::vector<double>> matrix;


class DataSet{
    /**
     * @brief DataSet class
     * 
     */
    private:
        /**
         * @param inputData is the predictors
         * @param inputLabels is the labels
         * @param datasetType is the type of dataset : "classification" or "regression"
         */
        matrix inputData;
        matrix inputLabels;
        std::string datasetType;
    public:
        /**
         * @brief Construct a new Data Set object
         * 
         * @param dataFilePath 
         * @param labelsFilePath 
         * @param datasetType 
         */
        DataSet(std::string dataFilePath, std::string labelsFilePath, std::string datasetType);
        
        /**
         * @brief get the input data 
         * 
         * @return matrix 
         */
        matrix getInputData();

        /**
         * @brief get the input labels 
         * 
         * @return matrix 
         */
        matrix getInputLabels();

        /**
         * @brief get the number of rows
         * 
         * @return the number of rows
         */
        unsigned int getnRows();

        /**
         * @brief get the type name of this dataset
         * 
         * @return type name 
         */
        std::string getType();

        /**
         * @brief import csv file int matrix
         * 
         * @param filePath 
         * @param matrix 
         */
        void importCsv(std::string filePath, matrix &matrix);
        
        /**
         * @brief shuffle the dataset
         * 
         */
        void shuffle();
};

#endif
