//
// Created by Kliment Serafimov on 2019-02-24.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DECISION_TREE_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DECISION_TREE_H

#include "Data.h"
#include "bit_signature.h"
#include "net.h"
#include "Header.h"
#include "firstOrderLearning.h"


typedef Data::unit_gate_type unit_gate_type;

pair<bit_signature, int> ensamble_teacher
(
        Data* latice,
        int num_teachers,
        net::parameters param,
        net::PriorityTrainReport (net::*training_f)(Data*, net::parameters),
        vector<int> choose_from_dimensions
)
{
    vector<int> votes(latice->numInputs, 0);

    for(int i = 0;i<num_teachers;i++)
    {
        net new_teacher;

        if( num_teachers >= 2 ||
            param.neural_net == NULL ||
            param.neural_net->numInputs != latice->numInputs ||
            param.neural_net->numOutputs != latice->numOutputs)
        {
            new_teacher = net(latice->numInputs, param.get_first_layer(latice->numInputs), latice->numOutputs);
        }
        else
        {
            new_teacher = *param.neural_net;
        }

        //new_teacher.set_special_weights();
        new_teacher.train(latice, param, training_f);

        int local_bit = new_teacher.the_model.get_worst_dimension(choose_from_dimensions);
        if(local_bit != -1)
        {
            //cout << "Some idea" <<endl;
            /*vector<int> active_bits = latice->get_active_bits();
            bool enter = false;
            for(int i = 0;i<active_bits.size();i++)
            {
                if(local_bit == active_bits[i])
                {
                    enter = true;
                }
            }
            assert(enter);*/
        }
        else
        {
            //cout << "no idea" <<endl;
            local_bit = latice->get_first_active_input_bit().vector_id;

            /*vector<int> active_bits = latice->get_active_bits();
            bool enter = false;
            for(int i = 0;i<active_bits.size();i++)
            {
                if(local_bit == active_bits[i])
                {
                    enter = true;
                }
            }
            assert(enter);*/
        }
        votes[local_bit]++;
    }

    pair<int, int> ret_bit = mp(-1, -1);

    sort_v(choose_from_dimensions);
    for(int at = 0; at<choose_from_dimensions.size();at++)
    {
        int i = choose_from_dimensions[at];
        //cout << votes[i] <<" ";
        ret_bit = max(ret_bit, mp(votes[i], i));
    }
    //cout << endl;


    /*cout << "at ensamble" <<endl;
    Data left_data, right_data;

    cout << latice->circuit[ret_bit.s].print() << " "<< ret_bit.f << endl;

    latice->split(latice->circuit[ret_bit.s], left_data, right_data);
    //left_data.remove_gates();
    //right_data.remove_gates();

    assert(left_data.size()>0);
    assert(right_data.size()>0);

    cout << left_data.size() <<" "<< right_data.size() <<endl;
    cout << "left" <<endl;
    for(int i = 0;i<left_data.size();i++)
    {
        cout << left_data.printInput(i)<<endl;;
    }
    cout << "right" <<endl;

    for(int i = 0;i<right_data.size();i++)
    {
        cout << right_data.printInput(i)<<endl;;
    }
    cout << "--"<<endl;*/

    return make_pair(latice->circuit[ret_bit.s], ret_bit.f);
}

pair<bit_signature, int> ensamble_teacher
(
        Data* latice,
        int num_teachers,
        net::parameters param,
        net::PriorityTrainReport (net::*training_f)(Data*, net::parameters)
)
{
    vector<int> choose_from_dimension;
    for(int i = 0;i<latice->numInputs;i++)
    {
        choose_from_dimension.push_back(i);
    }
    return ensamble_teacher(latice, num_teachers, param, training_f, choose_from_dimension);
}

pair<pair<bit_signature, bit_signature>, int>
ensamble_pair_teacher
(
        Data* latice,
        int num_teachers,
        net::parameters param,
        net::PriorityTrainReport (net::*training_f)(Data*, net::parameters),
        bool ret_vector,
        vector<net::data_model::bit_dimension_pair> &ret
)
{
    int get_top = 9;
    int n = latice->numInputs+1;
    vector<vector<int> > votes(latice->numInputs, vector<int>(latice->numInputs, 0));
    int vector_votes[n+1][n+1][get_top+1];
    memset(vector_votes, 0, sizeof(vector_votes));
    for(int i = 0;i<num_teachers;i++)
    {
        net new_teacher = net(latice->numInputs, latice->numOutputs);
        new_teacher.train(latice, param, training_f);

        vector<net::data_model::bit_dimension_pair> imporant_pair_dimensions = new_teacher.the_model.sort_dimension_pairs(latice);
        if(ret_vector)
        {
            for(int j = 0;j<imporant_pair_dimensions.size();j++)
            {
                if(imporant_pair_dimensions[0].val != 0)
                {
                    int f_dim = imporant_pair_dimensions[j].f_dim;
                    int s_dim = imporant_pair_dimensions[j].s_dim;
                    vector_votes[f_dim][s_dim][j]++;
                }
            }
        }

        if(imporant_pair_dimensions[0].val != 0)
        {
            int f_dim = imporant_pair_dimensions[0].f_dim;
            int s_dim = imporant_pair_dimensions[0].s_dim;
            votes[f_dim][s_dim]++;
        }
        else
        {
            assert(0);
            pair<bit_signature, bit_signature> p = latice->get_first_active_input_pair();
            votes[p.first.vector_id][p.second.vector_id]++;
        }
    }

    if(ret_vector)
    {
        vector<vector<int> > vis(latice->numInputs, vector<int>(latice->numInputs, 0));
        priority_queue<pair<pair<int, int>, pair<int, int> > > q;
        for(int i = 0;i<get_top; i++)
        {
            //pair<int, pair<int, int> > next = mp(-1, mp(-1, -1));

            for(int j = 0;j<latice->numInputs;j++)
            {
                for(int k = j+1; k<latice->numInputs;k++)
                {
                    if(vis[j][k]==0 && vector_votes[j][k][i]!=0)
                        q.push(mp(mp(get_top-i, vector_votes[j][k][i]), mp(j, k)));
                }
            }
            bool found = false;

            pair<pair<int, int>, pair<int, int> > next;
            while(!found)
            {
                assert(!q.empty());
                pair<pair<int, int>, pair<int, int> > at_next = q.top();
                q.pop();
                if(vis[at_next.s.f][at_next.s.s] == 0)
                {
                    next = at_next;
                    found = true;
                }
            }

            vis[next.s.f][next.s.s] = 1;
            //cout << next.s.f <<" "<< next.s.s <<" val = " << next.f.f <<" "<< next.f.s <<", ";
            ret.push_back(net::data_model::bit_dimension_pair({-1, -1, next.s.f, next.s.s, (double)(next.f.f<<16)+(next.f.s)}));
        }
        cout << endl;
    }

    pair<int, pair<int, int> > ret_bit;
    for(int i = 0; i<latice->numInputs;i++)
        for(int j = 0;j<latice->numInputs;j++)
        {
            //cout << votes[i] <<" ";
            ret_bit = max(ret_bit, mp(votes[i][j], make_pair(i, j)));
        }
    //cout << endl;

    return make_pair(mp(latice->circuit[ret_bit.s.f], latice->circuit[ret_bit.s.s]), ret_bit.f);
}

pair<pair<bit_signature, bit_signature>, int>
ensamble_pair_teacher
(
        Data* latice,
        int num_teachers,
        net::parameters param,
        net::PriorityTrainReport (net::*training_f)(Data*, net::parameters)
)
{
    vector<net::data_model::bit_dimension_pair> empty;
    return ensamble_pair_teacher(latice, num_teachers, param, training_f, false, empty);
}

class decision_tree
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

    decision_tree* left_child = NULL;
    decision_tree* right_child = NULL;

    decision_tree(){};

    decision_tree(Data* _original_data)
    {
        data_defined = true;
        original_data = *_original_data;
    }

    decision_tree(Data* _original_data, net::parameters training_parameters)
    {
        build_tree(_original_data, training_parameters);
    }

    void build_tree(Data* _original_data, net::parameters training_parameters)
    {
        original_data = *_original_data;
        data_defined = true;
        build_tree(training_parameters);
    }

    void build_tree(net::parameters training_parameters)
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
        left_child = new decision_tree(&left_data, training_parameters);
        print_close_local_data_model = false;
        right_child = new decision_tree(&right_data, training_parameters);
        print_close_local_data_model = true;
        print_close_local_data_model = prev_print_close_local_data_model;

        size = left_child->size+right_child->size+1;
        height = max(left_child->height, right_child->height)+1;

        //print_gate(0);
        //cout << endl;
    }

    void build_or_of_ands_circuit(net::parameters param)
    {
        net first_teacher = net(original_data.numInputs, original_data.numOutputs);
        first_teacher.train(&original_data, param, &net::softPriorityTrain);
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

    void neural_guided_node_selection(net::parameters param)
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
        //vector<vector<net> > net_for_branches;

        //original_data.printData("complete_data:");

        for(int i = 0;i<n;i++)
        {
            Data left, right;

            original_data.split_and_remove_bit(bit_signature(i), left, right);

            //second_order_data_for_branches.pb(vector<Data>());
            //second_order_data_for_branches[i].pb(left);
            //second_order_data_for_branches[i].pb(right);

            first_order_data_for_branches.pb(left);
            first_order_data_for_branches.pb(right);


            //cout << "split at i = " << i << endl;
            //left.printData("left:");
            //right.printData("right:");
        }

        firstOrderLearning<Data> meta_learner;
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
                net leaf_learner = param.progressive_ensamble_nets[n-1][j];
                //leaf_learner.printWeights();
                net::parameters leaf_parameters = cutoff_param(param.get_iteration_count(n - 1), 0.01);
                //cout << "init potential branch train" << endl;

                //leaf_learner.printWeights();

                leaf_learner.train(&first_order_data_for_branches[i], leaf_parameters, &net::softPriorityTrain);
                //cout << "end potential branch train" << endl;
                local_errors.pb(leaf_learner.get_error_of_data(&first_order_data_for_branches[i]).s);
            }
            per_dataset_errors.pb(local_errors);
        }

        //cout << "end with error calc, now agregate:" <<endl;

        pair<double, int> best_bit = mp(1000, -1);
        for(int i = 0, at_bit = 0;i<per_dataset_errors.size();i+=2, at_bit++)
        {
            vector<double> one = per_dataset_errors[i];
            assert(i+1 < per_dataset_errors.size());
            vector<double> two = per_dataset_errors[i+1];
            double size_one = param.ensamble_neural_model_error(n-1, one).size;
            double size_two = param.ensamble_neural_model_error(n-1, two).size;
            //cout << one[0] <<" " << size_one <<" | "<< two[0] << " " << size_two <<endl;
            double sum = size_one+size_two;

            best_bit = min(best_bit, mp(sum, at_bit));
        }

        //cout << "decided on bit = " << best_bit.s <<endl;
        augmented_data = original_data;
        decision_node = bit_signature(best_bit.s);
        assert(decision_node.vector_id == best_bit.s);
    }

    void confusion_guided_node_selection(net::parameters param)
    {
        Data local_data  = original_data;

        //NeuralNetworkWalker local_family_meta_trainer = NeuralNetworkWalker(local_data);
        //net first_teacher = local_family_meta_trainer.train_on_local_family();
        net first_teacher;
        if( param.neural_net == NULL ||
            param.neural_net->numInputs != local_data.numInputs ||
            param.neural_net->numOutputs != local_data.numOutputs)
        {
            first_teacher = net(local_data.numInputs, param.get_first_layer(local_data.numInputs), local_data.numOutputs);
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
                    important_pair_dimensions.push_back(mp(i, j));
                }
            }
        }

        //second_layer_circuit(important_pair_dimensions, param);
        select_decision_tree_node_by_trying_pairs_together(important_pair_dimensions, param);

    }

    void select_decision_tree_node_by_trying_pairs_together(vector<pair<int, int> > important_pair_dimensions, net::parameters param)
    {
        //cout << "prev second layer: " <<  augmented_data.numInputs << endl;
        original_data.apply_important_pair_dimensions_to_data(important_pair_dimensions, augmented_data);

        //cout << "num ins after second layer: " << second_layer_data.numInputs <<endl;
        vector<int> original_dimensions = original_data.get_active_bits();
        /*for(int i = 0;i<original_data.numInputs;i++)
        {
            original_dimensions.push_back(i);
        }*/
        decision_node = ensamble_teacher(&augmented_data, param.ensamble_size, param, &net::softPriorityTrain, original_dimensions).f;
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

    void second_layer_circuit(vector<pair<int, int> > imporant_pair_dimensions, net::parameters param)
    {
        original_data.apply_important_pair_dimensions_to_data(imporant_pair_dimensions, augmented_data);

        vector<net::data_model::bit_dimension_pair> second_order_imporant_pair_dimensions;
        ensamble_pair_teacher(&augmented_data, pow(param.ensamble_size, 2), param, &net::softPriorityTrain, true, second_order_imporant_pair_dimensions);

        select_decision_tree_node_by_trying_pairs_together(to_vec_pair_int_int(second_order_imporant_pair_dimensions), param);
    }

    vector<pair<int, int> > to_vec_pair_int_int(vector<net::data_model::bit_dimension_pair> imporant_pair_dimensions)
    {
        vector<pair<int, int> > operator_pair_ids;
        for(int i = 0;i<9;i++)
        {
            operator_pair_ids.push_back(make_pair(imporant_pair_dimensions[i].f_dim, imporant_pair_dimensions[i].s_dim));
        }
        return operator_pair_ids;
    }

    void construct_decision_tree_node(net::parameters param)
    {
        param.batch_width = 3;
        param.ensamble_size = 3;

        vector<net::data_model::bit_dimension_pair> imporant_pair_dimensions;
        ensamble_pair_teacher(&original_data, 20, param, &net::softPriorityTrain, true, imporant_pair_dimensions);

        //second_layer_circuit(to_vec_pair_int_int(imporant_pair_dimensions), param);
        //select_decision_tree_node_by_trying_pairs_together(to_vec_pair_int_int(imporant_pair_dimensions), param);
        select_decision_tree_node_by_trying_pairs_one_by_one(to_vec_pair_int_int(imporant_pair_dimensions), param);

    }



    void select_decision_tree_node_by_trying_pairs_one_by_one(vector<pair<int, int> > imporant_pair_dimensions, net::parameters param)
    {

        vector<operator_signature> operators_to_try = original_data.get_operators(imporant_pair_dimensions);

        /*

         pair<pair<bit_signature, bit_signature>, int>
         best_pair = ensamble_pair_teacher(&original_data, 20, training_parameters, &net::softPriorityTrain);

         vector<int> gate_operands;
         gate_operands.push_back(best_pair.f.f.vector_id);
         gate_operands.push_back(best_pair.f.s.vector_id);

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
                    the_bit_and_score = ensamble_teacher(&new_data, param.ensamble_size, param, &net::softPriorityTrain);

            bit_signature the_bit = the_bit_and_score.f;
            int score = the_bit_and_score.s;

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
            pair<bit_signature, int> tmp = ensamble_teacher(&original_data, param.ensamble_size, param, &net::softPriorityTrain);
            decision_node = tmp.f;
            augmented_data = original_data;
        }


    }

    void construct_decision_tree_node_by_trying_pairs_one_by_one(net::parameters param)
    {
        param.batch_width = 3;
        param.ensamble_size = 2;
        int new_operator_score = -1;
        bit_signature new_operator_id = -1;

        bool has_new_operator = false;

        Data* latice = &original_data;
        net new_teacher = net(latice->numInputs, latice->numOutputs);
        new_teacher.train(latice, param, &net::softPriorityTrain);

        vector<net::data_model::bit_dimension_pair> imporant_pair_dimensions = new_teacher.the_model.sort_dimension_pairs(latice);

        select_decision_tree_node_by_trying_pairs_one_by_one(to_vec_pair_int_int(imporant_pair_dimensions), param);



        //training_parameters.neutral_net = new net(original_data.numInputs, original_data.numOutputs);
        //training_parameters.neutral_net->train_to_neutral_data(&original_data, training_parameters);

        /*for(int test_id = 0; test_id < 10; test_id++)
        {
            pair<pair<bit_signature, bit_signature>, int> best_pair = ensamble_pair_teacher(&original_data, 60, training_parameters, &net::softPriorityTrain);
            cout << best_pair.f.f.vector_id <<" "<< best_pair.f.s.vector_id <<" "<< best_pair.s <<endl;

            //pair<bit_signature, int> the_bit_and_score = ensamble_teacher(&original_data, training_parameters.ensamble_size, training_parameters, &net::softPriorityTrain);

            //cout << the_bit_and_score.first.vector_id <<" "<< the_bit_and_score.second <<endl;
        }*/

        /*
        net first_teacher = net(original_data.numInputs, original_data.numOutputs);
        first_teacher.train(&original_data, training_parameters, &net::softPriorityTrain);*/

        /*vector<bit_signature> imporant_single_dimensions = first_teacher.the_model.sort_single_dimensions();
        vector<net::data_model::bit_dimension_pair> imporant_pair_dimensions = first_teacher.the_model.sort_dimension_pairs();
        vector<net::data_model::bit_dimension_pair> imporant_bit_pair_dimensions = first_teacher.the_model.sort_bit_dimension_pairs();

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

            pair<bit_signature, int> the_bit_and_score = ensamble_teacher(&new_data, training_parameters.ensamble_size, training_parameters, &net::softPriorityTrain);

            bit_signature the_bit = the_bit_and_score.f;
            int score = the_bit_and_score.s;

            int id = i;

            if(the_bit.vector_id == new_data.numInputs-1)
            {
                potential_operator_ids.push_back(mp(score, i));
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
            int id = potential_operator_ids[i].s;
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
