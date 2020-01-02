//
// Created by Kliment Serafimov on 2020-01-02.
//


#include "DecisionTreeSynthesisViaDP.h"


DecisionTreeScore _get_opt_decision_tree_score(int n, int func)
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