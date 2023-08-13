#ifndef DATA_SET_HPP
#define DATA_SET_HPP

#include <vector>
#include <string>

class DataSet{
    private:
        std::vector< std::vector <double>> inputData;
        std::vector< std::vector <double>> inputLabels;
    
    public:
        DataSet(std::string dataFilePath, std::string labelsFilePath);
        std::vector< std::vector <double>> getInputData();
        std::vector< std::vector <double>> getInputLabels();
        unsigned int getnRows();
        void importCsv(std::string filePath, std::vector< std::vector <double>> &matrix);
        void shuffle();
};

#endif
