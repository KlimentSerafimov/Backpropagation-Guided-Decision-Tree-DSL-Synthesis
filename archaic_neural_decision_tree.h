//
// Created by Kliment Serafimov on 2019-05-24.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_ARCHAIC_NEURAL_DECISION_TREE_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_ARCHAIC_NEURAL_DECISION_TREE_H

#include "Data.h"
#include "Header.h"

class archaic_neural_decision_tree
{
public:

    Data the_Data;
    net the_teacher;

    vector<pair<bit_signature, int> > gate;

    archaic_neural_decision_tree* left = NULL;
    archaic_neural_decision_tree* right = NULL;

    archaic_neural_decision_tree* new_left = NULL;
    archaic_neural_decision_tree* new_right = NULL;

    int original_size = 0;
    int new_size = 0;

    archaic_neural_decision_tree()
    {

    }


    int init_root(int _n, net::parameters param, net::PriorityTrainReport (net::*training_f)(Data*, net::parameters param))
    {
        //string type = "input_is_output";
        //string type = "longest_substring_double_to_middle";
        //string type = "longest_substring_of_two_strings";
        string type = "longestSubstring_ai_is_1";
        //string type = "vector_loop_language";
        //string type = "a_i_is_a_i_plus_one_for_all";
        //string type = "longestSubstring_dai_di_is_0";

        int inputSize = _n, outputSize;

        the_Data.generateData(inputSize, outputSize, type);
        int ret1 = build_tree(param, training_f);
        return ret1;
    }

    int simple_build_tree
    (
            net::parameters param,
            net::PriorityTrainReport (net::*training_f)(Data*, net::parameters)
    )
    {
        int inputSize = the_Data.numInputs;
        int outputSize = the_Data.numOutputs;
        int max_inter = inputSize;
        int hiddenLayers[10] =
                {
                        max_inter,
                        -1
                };

        the_teacher = net(inputSize, hiddenLayers, outputSize);

        //param.set_iteration_count(the_Data.size());

        bool expanded = false;

        original_size = 0;

        int ret = 0;
        int size_of_children = 0;

        assert(0);//check accuracy rate
        while(!the_teacher.test(&the_Data, 0.5).is_solved() && !expanded)
        {
            the_teacher.train(&the_Data, param, training_f);
            int the_bit = the_teacher.the_model.get_worst_dimension();

            if(the_bit != -1 || !the_Data.is_constant() /*|| !the_teacher.test(&the_Data, "no action")*/)
            {
                assert(ret == 0);
                expanded = true;

                if(the_bit == -1)
                {
                    the_bit = the_Data.get_first_active_input_bit();
                }

                gate.push_back(make_pair(the_bit, 1));

                left = new archaic_neural_decision_tree();
                right = new archaic_neural_decision_tree();

                the_Data.split(gate, left->the_Data, right->the_Data);



                assert(left->the_Data.size() > 0);
                assert(right->the_Data.size() > 0);

                bool enter = true;
                if(left->the_Data.size() >= 2)
                {
                    if(print_the_imporant_bit)cout << "l"<< the_bit <<endl;
                    size_of_children+=left->simple_build_tree(param, training_f);
                    original_size+=left->original_size;
                }
                else
                {
                    enter = false;
                    size_of_children+=1;
                    original_size++;
                }
                if(right->the_Data.size() >= 2)
                {
                    if(print_the_imporant_bit)cout << "r"<< the_bit <<endl;
                    size_of_children+=right->simple_build_tree(param, training_f);
                    original_size+=right->original_size;
                }
                else
                {
                    enter = false;
                    size_of_children+=1;
                    original_size++;
                }

            }
            else
            {
                if(print_the_imporant_bit)cout<<endl;
            }
            //cout << ret <<" "<< the_bit <<endl;
        }

        original_size+=1;
        ret+=1+size_of_children;
        return ret;

    }


    int build_tree(net::parameters param, net::PriorityTrainReport (net::*training_f)(Data*, net::parameters))
    {

        int improved = 0;
        int fail_to_improve = 0;
        int the_prev_bit = -1;

        int best_size = (1<<30);
        bit_signature best_bit;
        Data best_data = the_Data;
        bool is_constant = the_Data.is_constant();
        if(!is_constant)
        {
            for(int tries = 0, width = 3; tries<1;tries++)
            {
                Data local_data = the_Data;

                bit_signature the_bit;
                the_bit.vector_id = -1;

                if(false && local_data.num_active_input_bits()>1)
                {

                    Data local_local_data = local_data;
                    net first_teacher = net(local_local_data.numInputs, local_local_data.numOutputs);
                    first_teacher.train(&local_local_data, param, training_f);

                    typedef net::data_model::bit_dimension_pair bit_dimension_pair;
                    vector<bit_dimension_pair> sorted_pairs = first_teacher.the_model.sort_bit_dimension_pairs(&local_data); //first_teacher.the_model.sort_functional_dimension_pairs();//

                    int at = 0;
                    bool enter = false;
                    while(!enter && at < sorted_pairs.size())
                    {
                        bit_dimension_pair the_pair = sorted_pairs[at];
                        //cout << "Try :: :: " << the_pair.print() <<endl;
                        if(the_pair.val == 0)
                        {
                            break;
                        }
                        operator_signature new_operator = operator_signature(the_pair.f_bit, the_pair.s_bit, the_pair.f_dim, the_pair.s_dim);

                        if(print_try_and_finally)
                        {
                            cout << "Try :: :: " << the_pair.print() << endl;// << " " << bitset<4>(the_pair.the_gate.gate).to_string() << endl;
                        }
                        Data data_with_best_pair;
                        if(local_local_data.apply_new_operator_to_data(new_operator, data_with_best_pair))
                        {

                            operator_signature second_new_operator = operator_signature(!the_pair.f_bit, !the_pair.s_bit, the_pair.f_dim, the_pair.s_dim);

                            Data second_data_with_best_pair;
                            bool two_gates = false;
                            if(data_with_best_pair.apply_new_operator_to_data(second_new_operator, second_data_with_best_pair))
                            {
                                two_gates = true;
                                data_with_best_pair = second_data_with_best_pair;
                            }

                            net new_teacher = net(data_with_best_pair.numInputs, data_with_best_pair.numOutputs);
                            new_teacher.train(&data_with_best_pair, param, training_f);

                            int now_the_bit = new_teacher.the_model.get_worst_dimension();

                            if(now_the_bit == data_with_best_pair.numInputs-1 || now_the_bit == data_with_best_pair.numInputs-1-two_gates)
                            {
                                local_data = data_with_best_pair;

                                the_bit = local_data.circuit[now_the_bit];

                                Data data_with_single_kernel;
                                the_bit = local_data.add_single_kernel_to_base_and_discard_rest(the_bit, data_with_single_kernel);

                                local_data = data_with_single_kernel;



                                enter = true;
                                if(print_try_and_finally)
                                {
                                    cout << "Finally :: bit id =  " <<the_bit.vector_id<<" " << now_the_bit <<" : "<< the_pair.print() << " two gates: " << two_gates << endl;
                                }
                            }
                            else
                            {
                                //cout << "Fail :: :: " << the_pair.print() <<endl;
                                at++;
                            }
                        }
                        else
                        {
                            at++;
                        }
                    }

                }

                //if(the_bit.vector_id == -1)
                {
                    if(print_try_and_finally)
                    {
                        cout << "classic" <<endl;
                    }
                    net first_teacher = net(local_data.numInputs, local_data.numOutputs);
                    first_teacher.train(&local_data, param, training_f);
                    vector<bit_signature> bits_wanted = first_teacher.the_model.dimension_error_ratio_score;

                    for(int i = 0;i<bits_wanted.size();i++)
                    {
                        bits_wanted[i].vector_id = i;
                        double val = abs(bits_wanted[i].value);
                        if(isnan(val)) val = -1;
                        bits_wanted[i].set_sort_by(val);
                    }

                    sort_v(bits_wanted);
                    rev_v(bits_wanted);

                    Data data_with_all_kernels;
                    local_data.and_or_exstention(data_with_all_kernels, bits_wanted, width);
                    local_data = data_with_all_kernels;

                    the_bit = ensamble_teacher(&local_data, param.ensamble_size + 0*(local_data.numInputs/2+1), param, training_f).f;


                    Data data_with_single_kernel;
                    the_bit = local_data.add_single_kernel_to_base_and_discard_rest(the_bit, data_with_single_kernel);

                    local_data = data_with_single_kernel;

                }

                bit_signature the_local_final_bit = the_bit;

                int grown_children = 0;
                if(false)
                {
                    /*Data left_kernel, right_kernel;
                    gate.clear();
                    gate.push_back(mp(the_bit, 1));
                    local_data.split(gate, left_kernel, right_kernel);


                    assert(left_kernel.size() > 0);

                    assert(right_kernel.size() > 0);

                    archaic_neural_decision_tree* try_left = new archaic_neural_decision_tree();
                    archaic_neural_decision_tree* try_right = new archaic_neural_decision_tree();

                    //new_left = new archaic_neural_decision_tree();
                    //new_right = new archaic_neural_decision_tree();

                    try_left->the_Data = left_kernel;
                    try_right->the_Data = right_kernel;

                    //local_data.printData("here");

                    grown_children+=try_left->simple_build_tree(param, training_f);
                    grown_children+=try_right->simple_build_tree(param, training_f);
                     */
                }
                if(best_size >= grown_children)
                {
                    best_size = grown_children;

                    best_data = local_data;

                    best_bit = the_local_final_bit;
                }
                if(print_tree_synthesys)
                {
                    cout << local_data.size() << " ";
                    cout << " gate id: " << best_bit.vector_id;
                    if(best_bit.num_operators == 2)
                    {
                        cout << " " << bitset<4>(best_bit.gate).to_string() << "(" << best_bit.operands[0] <<", "<< best_bit.operands[1] <<")";

                    }
                    cout<< endl;
                }
            }

            Data left_kernel, right_kernel;
            gate.clear();
            gate.push_back(mp(best_bit, 1));
            best_data.split(gate, left_kernel, right_kernel);

            new_left = new archaic_neural_decision_tree();
            new_right = new archaic_neural_decision_tree();

            assert(left_kernel.size() > 0);

            assert(right_kernel.size() > 0);
            new_left->the_Data = left_kernel;
            new_right->the_Data = right_kernel;

            int grown_children = 0;
            grown_children+=new_left->build_tree(param, training_f);
            grown_children+=new_right->build_tree(param, training_f);

            the_Data = best_data;
        }
        else
        {
            assert(the_Data.size() > 0);
        }

        int grown_children = 0;
        //original_size = simple_build_tree(param, training_f);
        if(new_left != NULL && new_right != NULL)
        {
            grown_children = new_left->new_size + new_right->new_size;
        }
        else
        {
            grown_children = 0;
        }
        int ret = new_size = 1+grown_children;
        return ret;

    }

    void switch_off_synapse()
    {
        assert(0);
        int rand_layer = rand(0, the_teacher.size()-1);
        int rand_neuron = rand(0, the_teacher.layers[rand_layer].neurons.size()-1);
        int rand_disregard = rand(0, the_teacher.layers[rand_layer].neurons[rand_neuron].disregard.size()-1);
        int init_rand_neuron = rand_neuron;
        int init_rand_layer = rand_layer;
        while(false)//while not found what to switch off)
        {
            rand_neuron++;
            rand_neuron%=the_teacher.layers[rand_layer].size();
            if(rand_neuron == init_rand_neuron)
            {
                rand_layer++;
                rand_layer%=the_teacher.size();
                if(init_rand_layer == rand_layer)
                {
                    assert(0);
                }
            }
        }
        //damaged[rand_layer][rand_neuron] = true;
        //switch it off
        the_teacher.layers[rand_layer].neurons[rand_neuron].disregard[rand_disregard] = true;cout << "remove neuron in layer = " << rand_layer <<", neuron id ="<< rand_neuron <<" synapse id = "<< rand_disregard <<endl;

    }

    int disable_synapses
    (int max_inter, net::parameters param, net &the_teacher, Data &the_Data, net::PriorityTrainReport (net::*training_f)(Data*, net::parameters param))
    {
        vector<vector<bool> > damaged(the_teacher.size(), vector<bool>(max_inter, false));

        for(int i = 0;i<120;i++)
        {
            cout << "I " << i << endl;

            switch_off_synapse();

            assert(0);//check accuracy rate and test
            //the_teacher.test(&the_Data, "print result", 0.5);

            the_teacher.train(&the_Data, param, training_f);
        }

        return 1;
    }




    int print_gate(int t)
    {
        cout << indent(t);
        for(int i = 0, j = 0;i<the_Data.numInputs;i++)
        {
            if(j<gate.size())
            {
                if(gate[j].f.vector_id == i)
                {
                    if(gate[j].s == -1)
                    {
                        cout << "-";
                    }
                    else
                    {
                        assert(gate[j].s == 1);
                        cout << "+";
                    }
                    j++;
                }
                else
                {
                    cout << ".";
                }
            }
            else
            {
                cout << ".";
            }
        }
        if(gate.size() == 1)
        {
            bit_signature the_bit = gate[0].f;
            cout << " gate id: " << the_bit.vector_id;
            if(the_bit.num_operators == 2)
            {
                cout << " " << bitset<4>(the_bit.gate).to_string() << "(" << the_bit.operands[0] <<", "<< the_bit.operands[1] <<")";
            }
        }
        else
        {
            assert(gate.size() == 0);
        }
        cout << endl;
        bool enter = false;
        int ret = 1;
        if(new_left == NULL && new_right == NULL && left != NULL && right != NULL)
        {
            enter = true;
            ret+=left->print_gate(t+1);
            ret+=right->print_gate(t+1);
        }
        else if(new_left != NULL && new_right != NULL && left != NULL && right != NULL)
        {
            if(new_left->new_size + new_right->new_size > left->new_size+right->new_size)
            {
                //assert(0);
                enter = true;
                ret+=left->print_gate(t+1);
                ret+=right->print_gate(t+1);
            }
            else
            {
                enter = true;
                ret+=new_left->print_gate(t+1);
                ret+=new_right->print_gate(t+1);
            }
        }
        else
        {

            if(new_left != NULL && new_right != NULL && left == NULL && right == NULL)
            {
                enter = true;
                ret+=new_left->print_gate(t+1);
                ret+=new_right->print_gate(t+1);
            }
            else
            {
                assert(new_left == NULL && new_right == NULL && left == NULL && right == NULL);
            }
        }
        if(!enter)
        {
            for(int i = 0;i<the_Data.size();i++)
            {
                cout << indent(t) << the_Data.printInput(i) << " "<< the_Data.printOutput(i) << endl;
            }
        }
        cout << indent(t) << ret << " " << original_size+(!enter&((int)the_Data.size()<2)) << endl;
        return ret;
    }


    //int local_single_build(int _n, double min_rate, double max_rate , int parameter)
    net::PriorityTrainReport local_single_build(int _n, net::parameters param)
    {
        //string type = "input_is_output";
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

        the_teacher = net(inputSize, hiddenLayers, outputSize);
        return the_teacher.train(&the_Data, param, &net::softPriorityTrain);
        //assert(min_rate == max_rate);
        //return the_teacher.train(&the_Data, max_rate, &net::fullBatchTrain);
        //the_teacher.train(&the_Data, _rate, &net::hardPriorityTrain);
        //the_teacher.test(&the_Data, "print result");
        //the_teacher.analyzeMistakes(&the_Data);
    }
};

#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_ARCHAIC_NEURAL_DECISION_TREE_H
