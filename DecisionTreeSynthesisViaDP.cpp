//
// Created by Kliment Serafimov on 2020-01-02.
//


#include "DecisionTreeSynthesisViaDP.h"


DecisionTreeScore get_opt_decision_tree_score(int n, int func)
{
    //assert(0);
//    cout << "here?" <<endl;
//    cout << "buffer 2" <<endl;
    Data local_new_data = Data();
//    assert(0);
    local_new_data.init_exaustive_table_with_unary_output(n, func);

    DecisionTreeSynthesisViaDP<Data> decision_tree_solver;
    DecisionTreeScore size_opt;
    NeuralNetwork::parameters cutoff_parametes;
    size_opt = decision_tree_solver.synthesize_decision_tree_and_get_size
            (cutoff_parametes, n, local_new_data, optimal);

    return size_opt;
}

vector<FunctionAndDecisionTreeScore> get_smallest_f(int n)
{
    assert(n <= 4);
    if(n <= 3) {

        vector<int> fs;

        for (int i = 0; i < (1 << (1 << n)); i++) {
            fs.push_back(i);
        }

        vector<FunctionAndDecisionTreeScore> ordering;

        for (int i = 0; i < fs.size(); i++) {
//        cout << "here in get_smallest_f" <<endl;
            ordering.push_back(FunctionAndDecisionTreeScore(fs[i], get_opt_decision_tree_score(n, fs[i])));
        }

        //sort_v(ordering);
        return ordering;
    }

    else if(n == 4)
    {
        vector<int> fs;

        for (int i = 0; i < (1 << (1 << n)); i++) {
            if(10000*256.0/(1<<(1<<n)) > rand(0, 10000)) {
                fs.push_back(i);
            }
        }

        vector<FunctionAndDecisionTreeScore> ordering;

        for (int i = 0; i < fs.size(); i++) {
//        cout << "here in get_smallest_f" <<endl;
            ordering.push_back(FunctionAndDecisionTreeScore(fs[i], get_opt_decision_tree_score(n, fs[i])));
        }

        //sort_v(ordering);
        return ordering;
    }
    assert(false);
}