//
// Created by Kliment Serafimov on 2020-01-02.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_FIRSTORDERDATASET_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_FIRSTORDERDATASET_H

#include <vector>
#include <iostream>
#include "util.h"

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

    string print(int num_tabs)
    {
        string ret;
        ret+=get_tab(num_tabs)+"TRAIN\n";
        for(int i = 0;i<train_data.size();i++)
        {
            ret+=get_tab(num_tabs) + train_data[i].print(num_tabs+1)+"\n";
        }
        ret+=get_tab(num_tabs)+"TEST\n";
        for(int i = 0;i<test_data.size();i++)
        {
            ret+=get_tab(num_tabs) +test_data[i].print(num_tabs+1) +"\n";
        }
        return ret;
    }
};

#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_FIRSTORDERDATASET_H
