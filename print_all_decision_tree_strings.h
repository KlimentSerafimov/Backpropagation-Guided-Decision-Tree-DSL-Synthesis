//
// Created by Kliment Serafimov on 2020-01-02.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_PRINT_ALL_DECISION_TREE_STRINGS_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_PRINT_ALL_DECISION_TREE_STRINGS_H

#include <fstream>
#include "DecisionTreeScore.h"
#include "Data.h"
#include "DecisionTreeSynthesisViaDP.h"
#include "NeuralNetwork.h"


static ofstream fout_dataset("dataset.out");

string print_decision_tree_string(string decision_tree_string,  int num_tabs, char open, char close, bool do_print);

string print_decision_tree_string_python(string decision_tree_string,  int num_tabs, char open, char close);

string print_f(int n, int function);

string print_decision_tree_of_f_cpp(int n, int func);

//string print_decision_tree_of_f_python(int n, int func)
//{
//    string tabbed_string;
//    cout << "here in print_decision_tree_of_f_python" <<endl;
//    DecisionTreeScore score = get_opt_decision_tree_score(n, func);
//    assert(score.decision_tree_strings.size() == score.if_cpp_format_strings.size());
//    for(int i = 0;i<score.decision_tree_strings.size();i++) {
//        string local_str = score.if_python_format_strings[i];//"(dt=" + score.decision_tree_strings[i] + ")";
//        tabbed_string+=print_decision_tree_string_python(local_str, 0, '{', '}');
//        tabbed_string+="\n";
//        cout << "\n";
//    }
//    tabbed_string+="\n";
//    cout << "\n";
//    //fout_dataset << tabbed_string << endl;
//    return tabbed_string;
//}

void print_all_decision_tree_strings(int n);

#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_PRINT_ALL_DECISION_TREE_STRINGS_H
