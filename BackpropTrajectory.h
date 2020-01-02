//
// Created by Kliment Serafimov on 2019-05-20.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_BACKPROPTRAJECTORY_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_BACKPROPTRAJECTORY_H


#include "Header.h"
#include "Data.h"
#include "Batch.h"

class BackpropTrajectory: public vector<Batch> {

public:
    Data *origin;

    BackpropTrajectory(Data *_origin) {
        origin = _origin;
    }

    BackpropTrajectory()
    {

    }
    string print()
    {
        string ret;
        for(int i = 0; i<size();i++)
        {
            assert(at(i).elements.size() == 1); // can only print out batches of size 1
            ret+=std::to_string(at(i).elements[0].id) + " ";
        }
        return ret;
    }

};

#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_BACKPROPTRAJECTORY_H
