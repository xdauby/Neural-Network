#ifndef DATA_SET_HPP
#define DATA_SET_HPP

#include <vector>
#include <string>

class DataSet{
    private:
        std::vector< std::vector <double>> inputData;
        std::vector< std::vector <double>> inputLabels;
        std::string datasetType;
    public:
        DataSet(std::string dataFilePath, std::string labelsFilePath, std::string datasetType);
        std::vector< std::vector <double>> getInputData();
        std::vector< std::vector <double>> getInputLabels();
        unsigned int getnRows();
        std::string getType();
        void importCsv(std::string filePath, std::vector< std::vector <double>> &matrix);
        void shuffle();
};

#endif
