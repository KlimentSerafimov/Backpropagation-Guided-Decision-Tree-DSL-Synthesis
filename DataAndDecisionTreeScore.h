//
// Created by Kliment Serafimov on 2020-01-02.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DATAANDSCORE_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DATAANDSCORE_H

#include "Data.h"
#include "DecisionTreeScore.h"

class DataAndDecisionTreeScore: public Data
{
public:

    DecisionTreeScore score;

    DataAndDecisionTreeScore(DecisionTreeScore _score): Data()
    {
        score = _score;
    }

    DataAndDecisionTreeScore(): Data()
    {

    }


    virtual string print(int num_tabs)
    {

        return  printConcatinateOutput() + " " + score.print();

    }

    bool operator < (const DataAndDecisionTreeScore& other) const
    {
        return score < other.score;
    }

};


#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DATAANDSCORE_H
