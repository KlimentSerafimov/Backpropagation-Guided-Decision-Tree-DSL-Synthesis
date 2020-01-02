//
// Created by Kliment Serafimov on 2020-01-02.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DECISIONTREESCORE_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DECISIONTREESCORE_H

#include <vector>
#include <string>

using namespace std;

class DecisionTreeScore
{
public:
    double size = -1;
    int num_solutions = -1;

    vector<string> decision_tree_strings;
    vector<string> if_cpp_format_strings;
    vector<string> if_python_format_strings;


    DecisionTreeScore()
    {

    }

    DecisionTreeScore(double _size, string _decision_tree_in_string)
    {
        size = _size;
        decision_tree_strings.push_back(_decision_tree_in_string);
    }

    bool operator < (const DecisionTreeScore& other) const
    {
        /*if(size == other.size)
        {
            return num_solutions > other.num_solutions;
        }*/
        return size < other.size;
    }

    bool operator == (const DecisionTreeScore& other) const
    {
        return size == other.size && num_solutions == other.num_solutions;
    }

    string print()
    {
        return std::to_string((int)size);
        //return "("+std::to_string((int)size) + " " + std::to_string(num_solutions)+")";
    }

    string print(int id)
    {
        return "("+std::to_string((int)size) + + ", id=" + std::to_string(id)+")";
    }

};


#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DECISIONTREESCORE_H
