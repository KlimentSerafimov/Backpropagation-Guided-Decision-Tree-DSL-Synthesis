//
// Created by Kliment Serafimov on 2019-11-25.
//

#include "LanguagesOverBooleanFunctions.h"
#include "util.h"
#include <iostream>
#include <fstream>
#include <sstream>

#define MAX_NUM_PROGRAM_NODES 400000

static int global_num_program_nodes = 0;
vector<int> empty_slots__program_nodes;
ProgramNode all_program_nodes[MAX_NUM_PROGRAM_NODES];

ProgramNode* get_new_program_node(Primitive primitive)
{
    if(empty_slots__program_nodes.empty()) {
        if (global_num_program_nodes >= MAX_NUM_PROGRAM_NODES) {
            cout << "expand MAX_NUM_PROGRAM_NODES" << endl;
            assert(false);
        }
        all_program_nodes[global_num_program_nodes] = ProgramNode(global_num_program_nodes, primitive);
        return &all_program_nodes[global_num_program_nodes++];
    }
    else
    {
        int new_node_id = empty_slots__program_nodes.back();
        empty_slots__program_nodes.pop_back();
        all_program_nodes[new_node_id] = ProgramNode(new_node_id, primitive);
        return &all_program_nodes[new_node_id];
    }
}

void LanguagesOverBooleanFunctions::append_inputs(int order)
{
    assert(order == 1 || order == -1);
    if(order == 1) {
        for (int i = 0; i < num_inputs; i++) {
            primitives_by_num_inputs[0].push_back(Primitive(i, "x" + std::to_string(i)));
        }
    }
    else
    {
        for (int i = num_inputs-1; i >= 0; i--) {
            primitives_by_num_inputs[0].push_back(Primitive(i, "x" + std::to_string(i)));
        }
    }
}

LanguagesOverBooleanFunctions::LanguagesOverBooleanFunctions(int _num_inputs, int _language_id) {

    language_id = _language_id;
    num_inputs = _num_inputs;

    boolean_functions = vector<int>((1 << (1 << num_inputs)), 0);
    primitives_by_num_inputs = vector<vector<Primitive> >(3, vector<Primitive>());

    if (language_id == 0) {
        append_inputs(1);
        primitives_by_num_inputs[2].push_back(Primitive(PartialFunction(2, 1), "nor"));
    } else if (language_id == 1) {
        append_inputs(-1);
        primitives_by_num_inputs[2].push_back(Primitive(PartialFunction(2, 1), "nor"));
    } else if (language_id == 2) {
        append_inputs(1);
        primitives_by_num_inputs[2].push_back(Primitive(PartialFunction(2, 7), "nand"));
    } else if (language_id == 3) {
        append_inputs(-1);
        primitives_by_num_inputs[2].push_back(Primitive(PartialFunction(2, 7), "nand"));
    } else if (language_id == 4) {
        append_inputs(1);
        primitives_by_num_inputs[1].push_back(Primitive(PartialFunction(2, 1), "not"));
        primitives_by_num_inputs[2].push_back(Primitive(PartialFunction(2, 14), "or"));
    } else if (language_id == 5) {
        append_inputs(-1);
        primitives_by_num_inputs[1].push_back(Primitive(PartialFunction(2, 1), "not"));
        primitives_by_num_inputs[2].push_back(Primitive(PartialFunction(2, 14), "or"));
    } else if (language_id == 6) {
        append_inputs(1);
        primitives_by_num_inputs[1].push_back(Primitive(PartialFunction(2, 1), "not"));
        primitives_by_num_inputs[2].push_back(Primitive(PartialFunction(2, 8), "and"));
    } else if (language_id == 7) {
        append_inputs(-1);
        primitives_by_num_inputs[1].push_back(Primitive(PartialFunction(2, 1), "not"));
        primitives_by_num_inputs[2].push_back(Primitive(PartialFunction(2, 8), "and"));
    } else if (language_id == 8) {
        append_inputs(1);
        primitives_by_num_inputs[1].push_back(Primitive(PartialFunction(2, 1), "not"));
        primitives_by_num_inputs[2].push_back(Primitive(PartialFunction(2, 8), "and"));
        primitives_by_num_inputs[2].push_back(Primitive(PartialFunction(2, 14), "or"));
    } else if (language_id == 9) {
        append_inputs(-1);
        primitives_by_num_inputs[1].push_back(Primitive(PartialFunction(2, 1), "not"));
        primitives_by_num_inputs[2].push_back(Primitive(PartialFunction(2, 8), "and"));
        primitives_by_num_inputs[2].push_back(Primitive(PartialFunction(2, 14), "or"));
    } else if (language_id == 10) {

    } else if (language_id == 11) {

    } else if (language_id == 12) {

    } else {
        assert(false);
    }


    cout << "language = " << language_id << " :: ";

    for (int i = 0; i < primitives_by_num_inputs.size(); i++) {
        for (int j = 0; j < primitives_by_num_inputs[i].size(); j++) {
            cout << primitives_by_num_inputs[i][j].name << " ";
        }
    }
    cout << endl;
}

void LanguagesOverBooleanFunctions::enumerate() {

    if (language_id == 10) {
        assert(num_inputs == 3);
        int function_size = 8;
        //sum or minus
        int log_n = 2;
        int n = 4;
        int log_m = 2;
        int m = 4;
        int num_ops = 1;
        int op_log = 0;
        bucket_0 = 0;
        for(int i = 0;i<n;i++)
        {
            for(int j = 0;j<m;j++)
            {
                for(int k = 0;k<num_ops;k++)
                {
                    int sum_or_diff;
                    if(k == 0)
                    {
                        sum_or_diff = i+j;
                    }
                    else
                    {
                        sum_or_diff = i-j;
                        if(sum_or_diff < 0)
                        {
                            sum_or_diff = sum_or_diff*(-1);
                        }
                    }
                    int bits = i + (1<<log_n) * k + (1<<(log_n+op_log))*j + (1<<(log_n+op_log+log_m)) * sum_or_diff;
                    boolean_functions__insert(bits);
                    PartialFunction instance = PartialFunction(7, bits);
                    ordering_over_boolean_functions.push_back(instance);
                    bucket_0++;
                }
            }
        }
        for(int i = 0;i<(1<<7);i++)
        {
            if(!boolean_functions__contains(i))
            {
                ordering_over_boolean_functions.push_back(PartialFunction(function_size, i));
            }
        }
        return ;
    }

    if(num_inputs == 4)
    {
        ifstream fin;

        fin.open("ordering.in");

        string line;
        assert(fin.is_open());
        int i = 0;
        while(getline(fin, line))
        {
            int num;
            stringstream ss;
            ss << line;
            ss >> num;

//            cout << num << " ";
            ordering_over_boolean_functions.push_back(PartialFunction((1<<num_inputs), num));

        }
        cout << endl;
//        assert(false);
        return;

    }

    //size = 0
    program_roots_by_size.push_back(vector<ProgramNode*>());

    //size = 1
    program_roots_by_size.push_back(vector<ProgramNode*>());


    for(int i = 0;i<primitives_by_num_inputs[0].size();i++)
    {
        ProgramNode *new_program = get_new_program_node(primitives_by_num_inputs[0][i]);
        int total_function = get_total_function(num_inputs, new_program);

        if(!boolean_functions__contains(total_function))
        {
            boolean_functions__insert(total_function);
            ordering_over_boolean_functions.push_back(
                    PartialFunction((1<<num_inputs), total_function));
            program_roots_by_size[1].push_back(new_program);
//                                new_program->print();
        }else
        {
            new_program->my_delete();
        }
    }


    cout << "GENERATING LANGUAGE " << language_id << endl;
    //size >= 2
    int num_functions = (1<<(1<<num_inputs))-1;
    for(int size = 2; ordering_over_boolean_functions.size() < num_functions; size++)
    {
        program_roots_by_size.push_back(vector<ProgramNode*>());
        for(int local_num_inputs = 1; local_num_inputs < primitives_by_num_inputs.size(); local_num_inputs++)
        {
            assert(local_num_inputs <= 2);
            if(local_num_inputs == 1)
            {
                for(int i = 0;i<primitives_by_num_inputs[local_num_inputs].size();i++)
                {
                    for(int j = 0;j<program_roots_by_size[size-1].size();j++)
                    {
//                        if(program_roots_by_size[size-1][j]->is_constant())
//                        {
//                            continue;
//                        }

                        ProgramNode *new_program = get_new_program_node(primitives_by_num_inputs[local_num_inputs][i]);
                        new_program->inputs.push_back(program_roots_by_size[size-1][j]);
                        int total_function = get_total_function(num_inputs, new_program);

                        if(!boolean_functions__contains(total_function))
                        {
                            boolean_functions__insert(total_function);
                            ordering_over_boolean_functions.push_back(PartialFunction((1<<num_inputs), total_function));
                            program_roots_by_size[size].push_back(new_program);
//                                new_program->print();
                        }else
                        {
                            new_program->my_delete();
                        }
                    }
                }
            }
            else
            {
                for(int i = 0;i<primitives_by_num_inputs[local_num_inputs].size();i++)
                {
                    int sum_of_size_of_inputs = size-1;
                    for(int left_input_size = 1; left_input_size<sum_of_size_of_inputs;left_input_size++)
                    {
                        int right_input_size = sum_of_size_of_inputs - left_input_size;
//                        cout << left_input_size <<" "<< right_input_size << endl;
                        for(int j0 = 0;j0<program_roots_by_size[left_input_size].size();j0++)
                        {
                            for(int j1 = 0;j1<program_roots_by_size[right_input_size].size();j1++)
                            {
//                                if(program_roots_by_size[left_input_size][j0]->is_constant() &&
//                                        program_roots_by_size[right_input_size][j1]->is_constant())
//                                {
//                                    continue;
//                                }

                                ProgramNode *new_program = get_new_program_node(primitives_by_num_inputs[local_num_inputs][i]);
                                new_program->inputs.push_back(program_roots_by_size[left_input_size][j0]);
                                new_program->inputs.push_back(program_roots_by_size[right_input_size][j1]);

                                int total_function = get_total_function(num_inputs, new_program);
//                               todo: update_ordering_over_boolean_functions(size, generalization);

                                if(!boolean_functions__contains(total_function))
                                {
                                    boolean_functions__insert(total_function);
                                    ordering_over_boolean_functions.push_back(
                                            PartialFunction((1<<num_inputs), total_function));
                                    program_roots_by_size[size].push_back(new_program);
//                                new_program->print();
                                } else
                                {
                                    new_program->my_delete();
                                }

                            }
                        }
                    }
                }
            }
        }
//        cout << "size = " << size << " num_functions_discovered = "<< ordering_over_boolean_functions.size() << endl;
    }
    cout << "GENERATED TOTAL ORDER OF " << ordering_over_boolean_functions.size() << " FUNCITONS" << endl;


    bool do_print = true;

    if(do_print) {

        cout << "enumerate" << endl;
        cout << "language_" << language_id << " :: ";

        for (int i = 0; i < primitives_by_num_inputs.size(); i++) {
            for (int j = 0; j < primitives_by_num_inputs[i].size(); j++) {
                if (i != 0 || j != 0) {
                    cout << ", ";
                }
                cout << primitives_by_num_inputs[i][j].name;
            }
        }
        cout << " :: " << endl;

        int max_len = -1;
        for (int size = 0; size < program_roots_by_size.size(); size++) {
            for (int i = 0; i < program_roots_by_size[size].size(); i++) {
                string expr = program_roots_by_size[size][i]->to_string(0);
                max_len = max(max_len, (int) expr.size());
            }
        }

        string expr = "program";
        while (expr.size() < max_len) {
            expr = " " + expr;
        }
        cout << expr << " -> " << "outputs" << endl;

        for (int size = 0; size < program_roots_by_size.size(); size++) {
            for (int i = 0; i < program_roots_by_size[size].size(); i++) {
                int total_function = get_total_function(num_inputs, program_roots_by_size[size][i]);
                string expr = program_roots_by_size[size][i]->to_string(0);
                while (expr.size() < max_len) {
                    expr = " " + expr;
                }
                cout << expr << " -> " << bitvector_to_str(total_function, (1 << num_inputs)) << endl;

//            cout << total_function <<" ";
            }
        }
        cout << endl;
    }
}

MetaExample LanguagesOverBooleanFunctions::get_meta_example(PartialFunction partial_function) {
    for(int i = 0;i<ordering_over_boolean_functions.size();i++)
    {
        PartialFunction boolean_function = ordering_over_boolean_functions[i];
//        is_generalization_counter += __builtin_popcount(partial_function.partition);
        if(boolean_function.is_contained_in(partial_function))
        {
            return MetaExample(boolean_function.function_size, boolean_function.total_function, partial_function.partition);
        }
    }
    return MetaExample(partial_function.function_size, partial_function.total_function, partial_function.partition, partial_function.partition);
}

bool LanguagesOverBooleanFunctions::boolean_functions__contains(int function_id) {
    return boolean_functions[function_id] == 1;
}

void LanguagesOverBooleanFunctions::boolean_functions__insert(int function_id) {
    boolean_functions[function_id] = 1;
}

int get_total_function(int num_inputs, ProgramNode* program_node)
{
    if(!program_node->executed) {
        int total_function = 0;
//    program_node->print();
        int num_rows = (1 << num_inputs);
        for (int x = 0; x < num_rows; x++) {
            total_function += (program_node->execute(x) << x);
//        cout << bitvector_to_str(x, 2) << " " << result << endl;
        }
        program_node->total_function = total_function;
        program_node->executed = true;
        return total_function;
    }
    else
    {
        return program_node->total_function;
    }
}

Primitive::Primitive(PartialFunction _partial_function, string _name) {
    partial_function = _partial_function;
    primitive_type = operator_node;
    name = _name;
}

Primitive::Primitive(int _input_node_id, string _name) {
    input_node_id = _input_node_id;
    primitive_type = input_node;
    name = _name;
}

int Primitive::execute(int x) {
    if(primitive_type == input_node)
    {
        return get_bit(x, input_node_id);
    }
    else
    {
        return get_bit(partial_function.total_function, x);
    }
}

ProgramNode::ProgramNode(int _global_id, Primitive _primitive) {
    global_id = _global_id;
    primitive = _primitive;
}

string ProgramNode::to_string(int num_tabs) {
    string ret;
    ret +=
//            tabs(num_tabs) +
            primitive.name;
    if(inputs.size() >= 1)
    {
        ret+="(";
//        ret+="\n";
    }
    for(int i = 0;i<inputs.size();i++)
    {
        if(i!=0)
        {

            ret += ", ";
        }
        ret += inputs[i]->to_string(num_tabs+1);
//        ret += "\n";
    }
    if(inputs.size()>=1)
    {
        ret+=")";
//        ret+=tabs(num_tabs);
    }
    return ret;
}

void ProgramNode::print() {
    cout << to_string(0) <<endl;
}

bool ProgramNode::is_constant() {
    return primitive.primitive_type == operator_node && inputs.size() == 0;
}

int ProgramNode::execute(int x) {
    if(!executed) {
        if (inputs.size() >= 1) {
            vector<int> results;
            int root_x = 0;
            for (int i = 0; i < inputs.size(); i++) {
                root_x += (inputs[i]->execute(x) << i);
            }
            return primitive.execute(root_x);
        } else
        {
            return primitive.execute(x);
        }
    }
    else
    {
        return get_bit(total_function, x);
    }
}

void ProgramNode::my_delete() {
    empty_slots__program_nodes.push_back(global_id);
    inputs.clear();
}

