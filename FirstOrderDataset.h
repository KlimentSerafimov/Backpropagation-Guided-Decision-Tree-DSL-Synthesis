//
// Created by Kliment Serafimov on 2020-01-02.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_FIRSTORDERDATASET_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_FIRSTORDERDATASET_H

#include <vector>
#include <iostream>

using namespace std;

template<typename DataType>
class FirstOrderDataset
{
public:

    vector<DataType> train_data;
    vector<DataType> test_data;

    FirstOrderDataset()
    {

    }

    void print()
    {
        cout << "TRAIN" <<endl;
        for(int i = 0;i<train_data.size();i++)
        {
            cout << train_data[i].print() << endl;
        }
        cout << "TEST" <<endl;
        for(int i = 0;i<test_data.size();i++)
        {
            cout << test_data[i].print() << endl;
        }

    }
};

#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_FIRSTORDERDATASET_H
