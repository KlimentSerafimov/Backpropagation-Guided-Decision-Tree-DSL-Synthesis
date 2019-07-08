//
// Created by Kliment Serafimov on 2019-05-24.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_ARCHAIC_NEURAL_DESIGNED_CIRCUIT_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_ARCHAIC_NEURAL_DESIGNED_CIRCUIT_H

#include "archaic_neural_decision_tree.h"

class archaic_neural_designed_circuit
{
public:


    /*int build_circuit_based_on_pairs(int _n, net::parameters param, int (net::*training_f)(Data*, net::parameters))
     {
     Data the_Data;
        //string type = "input_is_output";
        //string type = "longest_substring_double_to_middle";
        //string type = "longest_substring_of_two_strings";
        string type = "longestSubstring_ai_is_1";
        //string type = "vector_loop_language";
        //string type = "a_i_is_a_i_plus_one_for_all";
        //string type = "longestSubstring_dai_di_is_0";

        int init_inputSize = _n, outputSize;

        the_Data.generateData(init_inputSize, outputSize, type);

        while(1)
        {
            net the_teacher = net(the_Data.numInputs, the_Data.numOutputs);
            the_teacher.train(&the_Data, param, training_f);

            typedef net::data_model::bit_dimension_pair bit_dimension_pair;
            vector<bit_dimension_pair> sorted_pairs = the_teacher.the_model.sort_bit_dimension_pairs();

            int at = 0;
            bool enter = false;
            while(!enter)
            {
                bit_dimension_pair the_pair = sorted_pairs[at];
                cout << "Try :: :: " << the_pair.print() <<endl;
                operator_signature new_operator = operator_signature(the_pair.f_bit, the_pair.s_bit, the_pair.f_dim, the_pair.s_dim);

                Data data_with_best_pair;
                if(the_Data.apply_new_operator_to_data(new_operator, data_with_best_pair))
                {

                    net new_teacher = net(data_with_best_pair.numInputs, data_with_best_pair.numOutputs);
                    new_teacher.train(&data_with_best_pair, param, training_f);

                    int the_bit = new_teacher.the_model.get_worst_dimension();
                    cout << the_bit <<endl;
                    if(the_bit == data_with_best_pair.numInputs-1)
                    {
                        enter = true;
                        cout << "Finally :: :: " << the_pair.print() <<endl;
                        the_Data = data_with_best_pair;
                    }
                    else
                    {
                        cout << "Fail :: :: " << the_pair.print() <<endl;
                        at++;
                    }
                }
                else
                {
                    at++;
                }
            }
        }

    }*/

    int build_circuit_based_on_singletons(int _n, net::parameters param, net::PriorityTrainReport (net::*training_f)(Data*, net::parameters))
    {
        Data the_Data;
        //string type = "input_is_output";
        //string type = "longest_substring_double_to_middle";
        //string type = "longest_substring_of_two_strings";
        string type = "longestSubstring_ai_is_1";
        //string type = "vector_loop_language";
        //string type = "a_i_is_a_i_plus_one_for_all";
        //string type = "longestSubstring_dai_di_is_0";

        int inputSize = _n, outputSize;

        the_Data.generateData(inputSize, outputSize, type);

        inputSize = the_Data.numInputs;
        outputSize = the_Data.numOutputs;
        int max_inter = inputSize;
        int hiddenLayers[10] =
                {
                        max_inter,
                        -1
                };

        //cout << the_Data.size() <<endl;
        net the_teacher = net(inputSize, hiddenLayers, outputSize);
        the_teacher.train(&the_Data, param, training_f);

        //vector<bit_signature> bits_wanted = the_teacher.the_model.bit_weighted_negative_dimension;
        vector<bit_signature> bits_wanted = the_teacher.the_model.dimension_error_ratio_score;


        for(int i = 0;i<bits_wanted.size();i++)
        {
            bits_wanted[i].vector_id = i;
            bits_wanted[i].set_sort_by(abs(bits_wanted[i].value));
        }

        sort_v(bits_wanted);
        rev_v(bits_wanted);

        for(int i = 0;i<bits_wanted.size();i++)
        {
            cout << bits_wanted[i].value <<"(" << bits_wanted[i].vector_id <<"), ";
        }
        cout << endl;

        //vector<Data> all_Datas;
        //all_Datas.push_back(the_Data);

        //vector<vector<bit_signature> > all_bits_wanted;
        //all_bits_wanted.push_back(bits_wanted);

        priority_queue<pair<bit_signature, int> > best_pairs;

        bool printDecisionTree = true;

        int trials = 1;
        if(printDecisionTree)
        {
            cout << "decision tree with no circuit: " <<endl;
            for(int i = 0;i<trials;i++)
            {
                archaic_neural_decision_tree special_root;
                special_root.the_Data = the_Data;
                cout << special_root.build_tree(param, training_f) <<"\t";
                special_root.print_gate(0);
            }
            cout << endl;
        }

        for(int i = 0;i<bits_wanted.size();i++)
        {
            for(int j = i+1; j<bits_wanted.size();j++)
            {
                //cout << "insert " << bits_wanted[i].get_sort_by()+bits_wanted[j].get_sort_by() <<endl;
                bit_signature first_operand = bit_signature(bits_wanted[i].get_sort_by()+bits_wanted[j].get_sort_by(), i, j);
                best_pairs.push(mp(first_operand, 0));
            }
        }

        while(!best_pairs.empty())
        {
            pair<bit_signature, int> circuit_data = best_pairs.top();
            best_pairs.pop();

            //Data* the_local_Data = &all_Datas[circuit_data.s];
            Data* the_local_Data = &the_Data;

            bit_signature at_pair = circuit_data.f;

            //vector<bit_signature> local_bits_wanted = all_bits_wanted[circuit_data.s];
            vector<bit_signature> local_bits_wanted = bits_wanted;//all_bits_wanted[circuit_data.s];

            int loc_1 = at_pair.operands[0], loc_2 = at_pair.operands[1];
            operator_signature new_operator = operator_signature((local_bits_wanted[loc_1]>0), (local_bits_wanted[loc_2]>0), local_bits_wanted[loc_1].vector_id, local_bits_wanted[loc_2].vector_id);

            Data data_with_best_pair;

            if(the_local_Data->apply_new_operator_to_data(new_operator, data_with_best_pair))
            {
                net new_teacher = net(data_with_best_pair.numInputs, data_with_best_pair.numOutputs);
                new_teacher.train(&data_with_best_pair, param, training_f);

                vector<bit_signature> new_bits_wanted = new_teacher.the_model.dimension_error_ratio_score;

                for(int i = 0;i<new_bits_wanted.size();i++)
                {
                    new_bits_wanted[i].vector_id = i;
                    new_bits_wanted[i].set_sort_by(abs(new_bits_wanted[i].value));
                }

                sort_v(new_bits_wanted);
                rev_v(new_bits_wanted);

                /*cout << at_pair.get_sort_by() << " " << loc_1 <<" " << loc_2 << " :: ";
                for(int i = 0;i<new_bits_wanted.size();i++)
                {
                    cout << new_bits_wanted[i].value <<"(" << new_bits_wanted[i].vector_id <<"), " ;
                }
                cout << endl;*/

                //if(new_bits_wanted[0].vector_id == new_bits_wanted.size()-1)
                {
                    int base_id = 0;
                    cout << data_with_best_pair.numInputs-data_with_best_pair.circuit.size() <<" vars in " << data_with_best_pair.circuit.size() <<" gates :: ";
                    for(int i = 0;i<data_with_best_pair.circuit.size();i++)
                    {
                        assert(data_with_best_pair.circuit[i].operands.size() == 2);
                        cout << "gate(" << bitset<4>(data_with_best_pair.circuit[i].gate).to_string() <<", "
                             << data_with_best_pair.circuit[i].operands[0] <<", "<< data_with_best_pair.circuit[i].operands[1] <<"); ";
                    }
                    cout << endl;

                    if(printDecisionTree)
                    {
                        cout << "decision tree with this circuit: ";
                        for(int i = 0;i<trials;i++)
                        {
                            archaic_neural_decision_tree special_root;
                            special_root.the_Data = data_with_best_pair;
                            cout << special_root.build_tree(param, training_f) <<endl;
                            special_root.print_gate(0);
                        }
                        cout << endl;
                    }

                    //all_Datas.push_back(data_with_best_pair);
                    the_Data = data_with_best_pair;
                    //all_bits_wanted.push_back(new_bits_wanted);
                    bits_wanted = new_bits_wanted;

                    while(!best_pairs.empty())
                    {
                        best_pairs.pop();
                    }

                    for(int i = 0;i<new_bits_wanted.size();i++)
                    {
                        for(int j = i+1; j<new_bits_wanted.size();j++)
                        {
                            if(new_bits_wanted[i].get_sort_by() > 0 && new_bits_wanted[j].get_sort_by() > 0 )
                            {
                                bit_signature first_operand = bit_signature(new_bits_wanted[i].get_sort_by()+new_bits_wanted[j].get_sort_by(), i, j);
                                best_pairs.push(mp(first_operand, -1));
                            }
                        }
                    }
                }
            }
        }
        return 1;
    }

    void single_expansion_step(Data the_data, net::parameters param, vector<operator_signature> &operators, set<pair<int, pair<int, int> > > &dp_exists, bool &enter)
    {
        int num_in = the_data.numInputs, num_out = the_data.numOutputs;
        net first_teacher = net(num_in, num_out);
        first_teacher.train(&the_data, param, &net::softPriorityTrain);

        vector<vector<bit_signature> > sorted_iouts = first_teacher.the_model.sort_inout_dimensions();


        //vector<operator_signature> operators;

        for(int i = 0;i<num_out;i++)
        {
            vector<pair<double, pair<int, int> > > sorted_local_gates;
            for(int j = 0;j<num_in;j++)
            {
                for(int k = j+1;k<num_in;k++)
                {
                    sorted_local_gates.push_back
                            (mp(sorted_iouts[i][j].value+sorted_iouts[i][k].value, mp(sorted_iouts[i][j].vector_id, sorted_iouts[i][k].vector_id)));
                }
            }
            sort_v(sorted_local_gates);
            rev_v(sorted_local_gates);

            vector<pair<int, int> > new_gates;

            int width = 10;
            for(int j = 0;j<min((int)sorted_local_gates.size(), width);j++)
            {
                new_gates.push_back(sorted_local_gates[j].s);
            }

            Data local_data;
            //the_data.apply_dnf_important_pair_dimensions_to_data(new_gates, local_data);
            the_data.apply_dnf_important_pair_dimensions_to_data(new_gates, local_data);

            net second_teacher = net(local_data.numInputs, local_data.numOutputs);
            second_teacher.train(&local_data, param, &net::softPriorityTrain);

            vector<vector<bit_signature> > single_dimensions = second_teacher.the_model.sort_inout_dimensions();

            int top_cutoff = 1;
            for(int j = 0;j<min(top_cutoff, (int)single_dimensions[i].size());j++)
            {
                int potential_id = single_dimensions[i][j].vector_id;
                if(potential_id >= num_in)
                {
                    pair<int, pair<int, int> > new_gate = mp(local_data.circuit[potential_id].gate, local_data.circuit[potential_id].operands_to_pair());
                    if(dp_exists.find(new_gate) == dp_exists.end())
                    {
                        enter = true;
                        dp_exists.insert(new_gate);
                        operators.push_back(local_data.circuit[potential_id]);
                    }
                }
            }
        }

    }

    int build_circuit_per_output_dimension(Data the_data, net::parameters param)
    {
        set<pair<int, pair<int, int> > > dp_exists;
        bool enter = true;
        int new_dimension_init = 0;
        int new_dimension_end = the_data.numInputs-1;
        while(enter)
        {
            enter = false;
            vector<operator_signature> operators;
            for(int i = new_dimension_init; i<=new_dimension_end; i++)
            {
                operator_signature gate = the_data.circuit[i];
                Data left, right;
                the_data.split(gate, left, right);

                single_expansion_step(left, param, operators, dp_exists, enter);
                single_expansion_step(right, param, operators, dp_exists, enter);

            }

            Data augmented_data;
            the_data.apply_new_operators_to_data(operators, augmented_data);
            the_data = augmented_data;

            new_dimension_init = new_dimension_end+1;
            new_dimension_end = the_data.numInputs-1;

            for(int i = 0;i<the_data.circuit.size();i++)
            {
                cout << the_data.circuit[i].print() <<endl;
            }
            cout << endl;

            //the_data.printData("data");
            int score = the_data.get_score(true);
            cout << "score = " << score << endl;
        }
        return 1;
    }
};

#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_ARCHAIC_NEURAL_DESIGNED_CIRCUIT_H
