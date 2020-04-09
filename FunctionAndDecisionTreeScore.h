//
// Created by Kliment Serafimov on 2020-01-02.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_FUNCTIONANDDECISIONTREESCORE_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_FUNCTIONANDDECISIONTREESCORE_H

#include "DecisionTreeScore.h"

class FunctionAndDecisionTreeScore: public DecisionTreeScore
{
public:
    long long function;

    FunctionAndDecisionTreeScore(int _function, int _size): DecisionTreeScore()
    {
        function = _function;
        size = _size;
        num_solutions = 1;
    }

    FunctionAndDecisionTreeScore(int _function, DecisionTreeScore tmp): DecisionTreeScore()
    {
        function = _function;
        size = tmp.size;
        num_solutions = tmp.num_solutions;
    }

    operator int()
    {
        return function;
    }
};


#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_FUNCTIONANDDECISIONTREESCORE_H
