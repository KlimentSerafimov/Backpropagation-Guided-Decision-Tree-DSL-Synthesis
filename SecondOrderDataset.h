//
// Created by Kliment Serafimov on 2020-01-02.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_SECONDORDERDATASET_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_SECONDORDERDATASET_H

#include <vector>

using namespace std;

template<typename DataType>
class SecondOrderDataset
{
public:
    int n;
    vector<vector<DataType> > train_meta_data;
    vector<vector<DataType> > test_meta_data;
};



#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_SECONDORDERDATASET_H
