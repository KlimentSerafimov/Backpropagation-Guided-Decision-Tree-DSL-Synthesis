//
// Created by Kliment Serafimov on 3/7/20.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_LANGUAGESOVERBOOLEANFUNCTIONS_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_LANGUAGESOVERBOOLEANFUNCTIONS_H


#include <vector>
#include <set>
#include "MetaExample.h"

using namespace std;

enum PrimitiveType {input_node, operator_node};

class Primitive
{
public:
    PrimitiveType primitive_type;
    string name;
    //if operator_node
    PartialFunction partial_function;
    //if input_node
    int input_node_id;

    Primitive() = default;
    Primitive(PartialFunction _partial_function, string _name);
    Primitive(int _input_node_id, string _name);

    int execute(int x);
};

class ProgramNode
{
public:

    int global_id;

    Primitive primitive;
    vector<ProgramNode*> inputs;

    int executed = false;
    int total_function;

    ProgramNode() = default;
    ProgramNode(int global_id, Primitive _primitive);

    string to_string(int num_tabs);
    void print();

    bool is_constant();

    int execute(int x);

    void my_delete();
};


class LanguagesOverBooleanFunctions {

    int language_id;
    int num_inputs;
    vector<vector<Primitive> > primitives_by_num_inputs;

    vector<vector<ProgramNode*> > program_roots_by_size;

//    set<int> boolean_functions;
    vector<int> boolean_functions;

    void append_inputs(int order);

public:
    int bucket_0;
    vector<PartialFunction> ordering_over_boolean_functions;

    int is_generalization_counter = 0;

    LanguagesOverBooleanFunctions(int num_inputs, int _language_id);

    void enumerate();

    MetaExample get_meta_example(PartialFunction partial_function);

    bool boolean_functions__contains(int function_id);

    void boolean_functions__insert(int function_id);

};

int get_total_function(int num_inputs, ProgramNode* program_node);



#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_LANGUAGESOVERBOOLEANFUNCTIONS_H
