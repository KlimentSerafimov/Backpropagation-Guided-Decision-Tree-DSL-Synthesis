//
// Created by Kliment Serafimov on 2019-02-24.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DECISION_TREE_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DECISION_TREE_H

#include "Data.h"
#include "bit_signature.h"
#include "NeuralNetwork.h"
#include "Header.h"
#include "FirstOrderLearning.h"


typedef Data::unit_gate_type unit_gate_type;

pair<bit_signature, int> ensamble_teacher
(
        Data* latice,
        int num_teachers,
        NeuralNetwork::parameters param,
        NeuralNetwork::PriorityTrainReport (NeuralNetwork::*training_f)(Data*, NeuralNetwork::parameters),
        vector<int> choose_from_dimensions
);

pair<bit_signature, int> ensamble_teacher
(
        Data* latice,
        int num_teachers,
        NeuralNetwork::parameters param,
        NeuralNetwork::PriorityTrainReport (NeuralNetwork::*training_f)(Data*, NeuralNetwork::parameters)
);

pair<pair<bit_signature, bit_signature>, int>
ensamble_pair_teacher
(
        Data* latice,
        int num_teachers,
        NeuralNetwork::parameters param,
        NeuralNetwork::PriorityTrainReport (NeuralNetwork::*training_f)(Data*, NeuralNetwork::parameters),
        bool ret_vector,
        vector<NeuralNetwork::data_model::bit_dimension_pair> &ret
);

pair<pair<bit_signature, bit_signature>, int>
ensamble_pair_teacher
(
        Data* latice,
        int num_teachers,
        NeuralNetwork::parameters param,
        NeuralNetwork::PriorityTrainReport (NeuralNetwork::*training_f)(Data*, NeuralNetwork::parameters)
);

class DecisionTree
{
public:
    bool data_defined = false;
    Data original_data;

    //after processing

    Data augmented_data;

    bool is_leaf = false;
    bit_signature decision_node;
    vector<bit_signature> decision_circuit;

    int height = 1;
    int size = 1;

    DecisionTree* left_child = NULL;
    DecisionTree* right_child = NULL;

    DecisionTree(){};

    DecisionTree(Data* _original_data)
    {
        data_defined = true;
        original_data = *_original_data;
    }

    DecisionTree(Data* _original_data, NeuralNetwork::parameters training_parameters)
    {
        build_tree(_original_data, training_parameters);
    }

    void build_tree(Data* _original_data, NeuralNetwork::parameters training_parameters)
    {
        original_data = *_original_data;
        data_defined = true;
        build_tree(training_parameters);
    }

    void build_tree(NeuralNetwork::parameters training_parameters)
    {
        assert(data_defined);
        if(original_data.is_constant())
        {
            is_leaf = true;
            augmented_data = original_data;
            return;
        }

        //construct_decision_tree_node_by_trying_pairs_one_by_one(training_parameters);

        if(training_parameters.decision_tree_synthesiser_type == confusion_guided)
        {
            confusion_guided_node_selection(training_parameters);
        }
        else if(training_parameters.decision_tree_synthesiser_type == neural_guided)
        {
            neural_guided_node_selection(training_parameters);
        }
        else if(training_parameters.decision_tree_synthesiser_type ==  entropy_guided)
        {
            entropy_split();
        }
        else if(training_parameters.decision_tree_synthesiser_type == combined_guided)
        {
            neural_guided_node_selection(training_parameters);
        }
        else if(training_parameters.decision_tree_synthesiser_type == random_guided)
        {
            random_split();
        }
        else
        {
            assert(0);
        }

        //construct_decision_tree_node(training_parameters);

        Data left_data, right_data;

        //cout << decision_node.print() <<endl;

        augmented_data.split_and_remove_bit(decision_node, left_data, right_data);
        //left_data.remove_gates();
        //right_data.remove_gates();

        assert(left_data.size()>0);
        assert(right_data.size()>0);
        assert(left_data.size() + right_data.size() == augmented_data.size());

        bool prev_print_close_local_data_model = print_close_local_data_model;
        print_close_local_data_model = false;
        left_child = new DecisionTree(&left_data, training_parameters);
        print_close_local_data_model = false;
        right_child = new DecisionTree(&right_data, training_parameters);
        print_close_local_data_model = true;
        print_close_local_data_model = prev_print_close_local_data_model;

        size = left_child->size+right_child->size+1;
        height = max(left_child->height, right_child->height)+1;

        //print_gate(0);
        //cout << endl;
    }

    void build_or_of_ands_circuit(NeuralNetwork::parameters param)
    {
        NeuralNetwork first_teacher = NeuralNetwork(original_data.numInputs, original_data.numOutputs);
        first_teacher.train(&original_data, param, &NeuralNetwork::softPriorityTrain);
    }

    void random_split()
    {
        srand(time(0));
        decision_node = bit_signature(rand(0, original_data.numInputs-1));
        augmented_data = original_data;
    }

    void entropy_split()
    {
        decision_node = original_data.get_most_entropic_input_bit();
        augmented_data = original_data;
    }

    void neural_guided_node_selection(NeuralNetwork::parameters param)
    {
        int n = original_data.numInputs;

        //cout << "ENTER neural_guided_node_selection n = " << n << endl;

        if(n == 1)
        {
            //cout << "at base case" <<endl;
            augmented_data = original_data;
            decision_node = bit_signature(0);
            assert(decision_node.vector_id == 0);
            return;
        }

        vector<Data> first_order_data_for_branches;
        //vector<vector<Data> > second_order_data_for_branches;
        //vector<vector<NeuralNetwork> > net_for_branches;

        //original_data.printData("complete_data:");

        for(int i = 0;i<n;i++)
        {
            Data left, right;

            original_data.split_and_remove_bit(bit_signature(i), left, right);

            //second_order_data_for_branches.push_back(vector<Data>());
            //second_order_data_for_branches[i].push_back(left);
            //second_order_data_for_branches[i].push_back(right);

            first_order_data_for_branches.push_back(left);
            first_order_data_for_branches.push_back(right);


            //cout << "split at i = " << i << endl;
            //left.printData("left:");
            //right.printData("right:");
        }

        FirstOrderLearning<Data> meta_learner;
        int ensamble_size = param.ensamble_size;
        //cout << "init reptile train for possible branches" <<endl;

        ///meta learn first
        //meta_learner.reptile_train(meta_net, first_order_data_for_branches,
        //        param.get_meta_iteration_count(), param.get_iteration_count(), 0.01);

        //cout << "end reptile train for possible branches" <<endl;

        vector<vector<double> > per_dataset_errors;
        for(int i = 0;i<first_order_data_for_branches.size();i++)
        {
            vector<double> local_errors;
            for(int j = 0; j < ensamble_size; j++)
            {
                assert(n-1<param.progressive_ensamble_nets.size());
                assert(j<param.progressive_ensamble_nets[n-1].size());
                NeuralNetwork leaf_learner = param.progressive_ensamble_nets[n-1][j];
                //leaf_learner.printWeights();
                NeuralNetwork::parameters leaf_parameters =
                        cutoff_param(param.get_iteration_count(n - 1), 0.01);
                //cout << "init potential branch train" << endl;

                //leaf_learner.printWeights();

                leaf_learner.train(&first_order_data_for_branches[i], leaf_parameters, &NeuralNetwork::softPriorityTrain);
                //cout << "end potential branch train" << endl;
                local_errors.push_back(leaf_learner.get_error_of_data(&first_order_data_for_branches[i]).second);
            }
            per_dataset_errors.push_back(local_errors);
        }

        //cout << "end with error calc, now agregate:" <<endl;

        pair<double, int> best_bit = make_pair(1000, -1);
        for(int i = 0, at_bit = 0;i<per_dataset_errors.size();i+=2, at_bit++)
        {
            vector<double> one = per_dataset_errors[i];
            assert(i+1 < per_dataset_errors.size());
            vector<double> two = per_dataset_errors[i+1];
            double size_one = param.ensamble_neural_model_error(n-1, one).size;
            double size_two = param.ensamble_neural_model_error(n-1, two).size;
            //cout << one[0] <<" " << size_one <<" | "<< two[0] << " " << size_two <<endl;
            double sum = size_one+size_two;

            best_bit = min(best_bit, make_pair(sum, at_bit));
        }

        //cout << "decided on bit = " << best_bit.second <<endl;
        augmented_data = original_data;
        decision_node = bit_signature(best_bit.second);
        assert(decision_node.vector_id == best_bit.second);
    }

    void confusion_guided_node_selection(NeuralNetwork::parameters param)
    {
        Data local_data  = original_data;

        //NeuralNetworkWalker local_family_meta_trainer = NeuralNetworkWalker(local_data);
        //NeuralNetwork first_teacher = local_family_meta_trainer.train_on_local_family();
        NeuralNetwork first_teacher;
        if( param.neural_net == NULL ||
            param.neural_net->numInputs != local_data.numInputs ||
            param.neural_net->numOutputs != local_data.numOutputs)
        {
            first_teacher = NeuralNetwork(local_data.numInputs, param.get_first_layer(local_data.numInputs), local_data.numOutputs);
        }
        else
        {
            first_teacher = *param.neural_net;
        }
        //first_teacher.set_special_weights();

        assert(param.track_dimension_model);
        first_teacher.train(&local_data, param, param.priority_train_f);
        //cout << "end init train" << endl;

        vector<bit_signature> bits_wanted = first_teacher.the_model.sort_single_dimensions();

        /*for(int i = 0;i<bits_wanted.size();i++)
        {
            cout << bits_wanted[i].vector_id <<" "<< bits_wanted[i].value << "; ";
        }
        cout << endl;*/

        vector<pair<int, int> > important_pair_dimensions;

        int width = 0; //!!! width is 0
        for(int i = 0;i<width;i++)
        {
            for(int j = i+1; j<width;j++)
            {
                if(bits_wanted[i].value != 0 && bits_wanted[j].value != 0)
                {
                    important_pair_dimensions.push_back(make_pair(i, j));
                }
            }
        }

        //second_layer_circuit(important_pair_dimensions, param);
        select_decision_tree_node_by_trying_pairs_together(important_pair_dimensions, param);

    }

    void select_decision_tree_node_by_trying_pairs_together(vector<pair<int, int> > important_pair_dimensions, NeuralNetwork::parameters param)
    {
        //cout << "prev second Layer: " <<  augmented_data.numInputs << endl;
        original_data.apply_important_pair_dimensions_to_data(important_pair_dimensions, augmented_data);

        //cout << "num ins after second Layer: " << second_layer_data.numInputs <<endl;
        vector<int> original_dimensions = original_data.get_active_bits();
        /*for(int i = 0;i<original_data.numInputs;i++)
        {
            original_dimensions.push_back(i);
        }*/
        decision_node = ensamble_teacher(&augmented_data, param.ensamble_size, param, &NeuralNetwork::softPriorityTrain, original_dimensions).first;
        assert(decision_node.vector_id < original_data.numInputs);
        //decision_node = augmented_data.select_most_confusing_dimension();
        //cout << "decision node : " << decision_node.print() <<endl <<endl;;

        /*for(int i = 0;i<augmented_data.circuit.size(); i++)
        {
            cout << augmented_data.circuit[i].print() <<"; ";
        }
        augmented_data = second_layer_data;*/

        Data data_with_single_kernel;

        decision_node = augmented_data.add_single_kernel_to_base_and_discard_rest(decision_node, data_with_single_kernel);
        augmented_data = data_with_single_kernel;

    }

    void second_layer_circuit(vector<pair<int, int> > imporant_pair_dimensions, NeuralNetwork::parameters param)
    {
        original_data.apply_important_pair_dimensions_to_data(imporant_pair_dimensions, augmented_data);

        vector<NeuralNetwork::data_model::bit_dimension_pair> second_order_imporant_pair_dimensions;
        ensamble_pair_teacher(&augmented_data, pow(param.ensamble_size, 2), param, &NeuralNetwork::softPriorityTrain, true, second_order_imporant_pair_dimensions);

        select_decision_tree_node_by_trying_pairs_together(to_vec_pair_int_int(second_order_imporant_pair_dimensions), param);
    }

    vector<pair<int, int> > to_vec_pair_int_int(vector<NeuralNetwork::data_model::bit_dimension_pair> imporant_pair_dimensions)
    {
        vector<pair<int, int> > operator_pair_ids;
        for(int i = 0;i<9;i++)
        {
            operator_pair_ids.push_back(make_pair(imporant_pair_dimensions[i].f_dim, imporant_pair_dimensions[i].s_dim));
        }
        return operator_pair_ids;
    }

    void construct_decision_tree_node(NeuralNetwork::parameters param)
    {
        param.batch_width = 3;
        param.ensamble_size = 3;

        vector<NeuralNetwork::data_model::bit_dimension_pair> imporant_pair_dimensions;
        ensamble_pair_teacher(&original_data, 20, param, &NeuralNetwork::softPriorityTrain, true, imporant_pair_dimensions);

        //second_layer_circuit(to_vec_pair_int_int(imporant_pair_dimensions), param);
        //select_decision_tree_node_by_trying_pairs_together(to_vec_pair_int_int(imporant_pair_dimensions), param);
        select_decision_tree_node_by_trying_pairs_one_by_one(to_vec_pair_int_int(imporant_pair_dimensions), param);

    }



    void select_decision_tree_node_by_trying_pairs_one_by_one(vector<pair<int, int> > imporant_pair_dimensions, NeuralNetwork::parameters param)
    {

        vector<operator_signature> operators_to_try = original_data.get_operators(imporant_pair_dimensions);

        /*

         pair<pair<bit_signature, bit_signature>, int>
         best_pair = ensamble_pair_teacher(&original_data, 20, training_parameters, &NeuralNetwork::softPriorityTrain);

         vector<int> gate_operands;
         gate_operands.push_back(best_pair.f.f.vector_id);
         gate_operands.push_back(best_pair.f.second.vector_id);

         vector<operator_signature> operators_to_try = original_data.get_operators(gate_operands);
         */


        int new_operator_score = -1;
        bit_signature new_operator_id = -1;

        bool has_new_operator = false;


        for(int i = 0;i<operators_to_try.size();i++)
        {
            Data new_data;
            original_data.apply_new_operator_to_data(operators_to_try[i], new_data);

            //cout <<"operator applied: " << operators_to_try[i].print() <<endl;
            pair<bit_signature, int>
                    the_bit_and_score = ensamble_teacher(&new_data, param.ensamble_size, param, &NeuralNetwork::softPriorityTrain);

            bit_signature the_bit = the_bit_and_score.first;
            int score = the_bit_and_score.second;

            int id = i;

            if(the_bit.vector_id == new_data.numInputs-1)
            {
                has_new_operator = true;
                if(new_operator_score < score)
                {
                    new_operator_score = score;
                    new_operator_id = the_bit;
                }
            }
        }

        if(has_new_operator)
        {
            decision_node = new_operator_id;
            original_data.apply_new_operator_to_data(decision_node, augmented_data);
        }
        else
        {
            pair<bit_signature, int> tmp = ensamble_teacher(&original_data, param.ensamble_size, param, &NeuralNetwork::softPriorityTrain);
            decision_node = tmp.first;
            augmented_data = original_data;
        }


    }

    void construct_decision_tree_node_by_trying_pairs_one_by_one(NeuralNetwork::parameters param)
    {
        param.batch_width = 3;
        param.ensamble_size = 2;
        int new_operator_score = -1;
        bit_signature new_operator_id = -1;

        bool has_new_operator = false;

        Data* latice = &original_data;
        NeuralNetwork new_teacher = NeuralNetwork(latice->numInputs, latice->numOutputs);
        new_teacher.train(latice, param, &NeuralNetwork::softPriorityTrain);

        vector<NeuralNetwork::data_model::bit_dimension_pair> imporant_pair_dimensions = new_teacher.the_model.sort_dimension_pairs(latice);

        select_decision_tree_node_by_trying_pairs_one_by_one(to_vec_pair_int_int(imporant_pair_dimensions), param);



        //training_parameters.neutral_net = new NeuralNetwork(original_data.numInputs, original_data.numOutputs);
        //training_parameters.neutral_net->train_to_neutral_data(&original_data, training_parameters);

        /*for(int test_id = 0; test_id < 10; test_id++)
        {
            pair<pair<bit_signature, bit_signature>, int> best_pair = ensamble_pair_teacher(&original_data, 60, training_parameters, &NeuralNetwork::softPriorityTrain);
            cout << best_pair.f.f.vector_id <<" "<< best_pair.f.second.vector_id <<" "<< best_pair.second <<endl;

            //pair<bit_signature, int> the_bit_and_score = ensamble_teacher(&original_data, training_parameters.ensamble_size, training_parameters, &NeuralNetwork::softPriorityTrain);

            //cout << the_bit_and_score.first.vector_id <<" "<< the_bit_and_score.second <<endl;
        }*/

        /*
        NeuralNetwork first_teacher = NeuralNetwork(original_data.numInputs, original_data.numOutputs);
        first_teacher.train(&original_data, training_parameters, &NeuralNetwork::softPriorityTrain);*/

        /*vector<bit_signature> imporant_single_dimensions = first_teacher.the_model.sort_single_dimensions();
        vector<NeuralNetwork::data_model::bit_dimension_pair> imporant_pair_dimensions = first_teacher.the_model.sort_dimension_pairs();
        vector<NeuralNetwork::data_model::bit_dimension_pair> imporant_bit_pair_dimensions = first_teacher.the_model.sort_bit_dimension_pairs();

        vector<operator_signature> operators_to_try = original_data.get_operators(range_vector(0, original_data.numInputs-1));

        vector<pair<int, int> > potential_operator_ids;
        vector<pair<bit_signature, Data> > potential_bits;

        for(int i = 0; i< imporant_single_dimensions.size(); i++)
        {
            cout << imporant_single_dimensions[i].vector_id << " "<< imporant_single_dimensions[i].value << endl;
        }

        for(int i = 0;i<imporant_bit_pair_dimensions.size();i++)
        {
            cout << imporant_bit_pair_dimensions[i].print() << endl;
        }
        cout << endl;
        for(int i = 0;i<imporant_pair_dimensions.size();i++)
        {
            cout << imporant_pair_dimensions[i].print() << endl;
        }
        cout << endl;

        for(int i = 0;i<operators_to_try.size();i++)
        {
            Data new_data;
            original_data.apply_new_operator_to_data(operators_to_try[i], new_data);

            pair<bit_signature, int> the_bit_and_score = ensamble_teacher(&new_data, training_parameters.ensamble_size, training_parameters, &NeuralNetwork::softPriorityTrain);

            bit_signature the_bit = the_bit_and_score.f;
            int score = the_bit_and_score.second;

            int id = i;

            if(the_bit.vector_id == new_data.numInputs-1)
            {
                potential_operator_ids.push_back(make_pair(score, i));
                cout << i <<endl;
                cout << bitset<4>(operators_to_try[id].gate).to_string() <<" " << operators_to_try[id].operands[0] << " "<< operators_to_try[id].operands[1] << " " << score <<endl;
                cout << "YES" <<endl;
            }
            else
            {
                //cout << "NO" <<endl;
            }
            potential_bits.push_back(make_pair(the_bit, new_data));
            //cout << endl;
        }

        sort_v(potential_operator_ids);

        for(int i = 0;i<potential_operator_ids.size();i++)
        {
            int id = potential_operator_ids[i].second;
            cout << bitset<4>(operators_to_try[id].gate).to_string() <<" " << operators_to_try[id].operands[0] << " "<< operators_to_try[id].operands[1] << " "<< potential_operator_ids[i].f << endl;
        }


        assert(0);*/

    }


    string indent(int n)
    {
        string s = "    ";
        string ret = "";
        for(int i = 0;i<n;i++)
        {
            ret+=s;
        }
        return ret;
    }

    void print_gate(int t)
    {
        cout << indent(t);
        if(!is_leaf)
        {

            for(int i = 0, j = 0;i<augmented_data.numInputs;i++)
            {
                if(decision_node.vector_id == i)
                {
                    cout << "+";
                }
                else
                {
                    cout << ".";
                }

            }
            cout << " gate: " << decision_node.print() << "; (s, h)=(" << size <<", " << height <<")"<< endl;bool enter = false;
            if(left_child!= NULL)
            {
                enter = true;
                left_child->print_gate(t+1);
            }
            if(right_child!= NULL)
            {
                enter = true;
                right_child->print_gate(t+1);
            }
            assert(enter == !is_leaf);
        }
        else
        {

            cout <<"leaf " << original_data.size() <<endl;

            for(int i = 0;i<augmented_data.size();i++)
            {
                cout << indent(t) << augmented_data.printInput(i) <<" "<< augmented_data.printOutput(i) <<endl;
            }
        }

    }


    vector<int> range_vector(int init, int end)
    {
        vector<int> ret;
        for(int i = init; i<=end; i++)
        {
            ret.push_back(i);
        }
        return ret;
    }
};

#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DECISION_TREE_H
