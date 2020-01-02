//
// Created by Kliment Serafimov on 2020-01-02.
//

#include "DecisionTree.h"

pair<bit_signature, int> ensamble_teacher
        (
                Data* latice,
                int num_teachers,
                NeuralNetwork::parameters param,
                NeuralNetwork::PriorityTrainReport (NeuralNetwork::*training_f)(Data*, NeuralNetwork::parameters),
                vector<int> choose_from_dimensions
        )
{
    vector<int> votes(latice->numInputs, 0);

    for(int i = 0;i<num_teachers;i++)
    {
        NeuralNetwork new_teacher;

        if( num_teachers >= 2 ||
            param.neural_net == NULL ||
            param.neural_net->numInputs != latice->numInputs ||
            param.neural_net->numOutputs != latice->numOutputs)
        {
            new_teacher = NeuralNetwork(latice->numInputs, param.get_first_layer(latice->numInputs), latice->numOutputs);
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

    pair<int, int> ret_bit = make_pair(-1, -1);

    sort_v(choose_from_dimensions);
    for(int at = 0; at<choose_from_dimensions.size();at++)
    {
        int i = choose_from_dimensions[at];
        //cout << votes[i] <<" ";
        ret_bit = max(ret_bit, make_pair(votes[i], i));
    }
    //cout << endl;


    /*cout << "at ensamble" <<endl;
    Data left_data, right_data;

    cout << latice->circuit[ret_bit.second].print() << " "<< ret_bit.f << endl;

    latice->split(latice->circuit[ret_bit.second], left_data, right_data);
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

    return make_pair(latice->circuit[ret_bit.second], ret_bit.first);
}

pair<bit_signature, int> ensamble_teacher
        (
                Data* latice,
                int num_teachers,
                NeuralNetwork::parameters param,
                NeuralNetwork::PriorityTrainReport (NeuralNetwork::*training_f)(Data*, NeuralNetwork::parameters)
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
                NeuralNetwork::parameters param,
                NeuralNetwork::PriorityTrainReport (NeuralNetwork::*training_f)(Data*, NeuralNetwork::parameters),
                bool ret_vector,
                vector<NeuralNetwork::data_model::bit_dimension_pair> &ret
        )
{
    int get_top = 9;
    int n = latice->numInputs+1;
    vector<vector<int> > votes(latice->numInputs, vector<int>(latice->numInputs, 0));
    int vector_votes[n+1][n+1][get_top+1];
    memset(vector_votes, 0, sizeof(vector_votes));
    for(int i = 0;i<num_teachers;i++)
    {
        NeuralNetwork new_teacher = NeuralNetwork(latice->numInputs, latice->numOutputs);
        new_teacher.train(latice, param, training_f);

        vector<NeuralNetwork::data_model::bit_dimension_pair> imporant_pair_dimensions = new_teacher.the_model.sort_dimension_pairs(latice);
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
            //pair<int, pair<int, int> > next = make_pair(-1, make_pair(-1, -1));

            for(int j = 0;j<latice->numInputs;j++)
            {
                for(int k = j+1; k<latice->numInputs;k++)
                {
                    if(vis[j][k]==0 && vector_votes[j][k][i]!=0)
                        q.push(make_pair(make_pair(get_top-i, vector_votes[j][k][i]), make_pair(j, k)));
                }
            }
            bool found = false;

            pair<pair<int, int>, pair<int, int> > next;
            while(!found)
            {
                assert(!q.empty());
                pair<pair<int, int>, pair<int, int> > at_next = q.top();
                q.pop();
                if(vis[at_next.second.first][at_next.second.second] == 0)
                {
                    next = at_next;
                    found = true;
                }
            }

            vis[next.second.first][next.second.second] = 1;
            //cout << next.second.f <<" "<< next.second.second <<" val = " << next.f.f <<" "<< next.f.second <<", ";
            ret.push_back(NeuralNetwork::data_model::bit_dimension_pair({-1, -1, next.second.first, next.second.second, (double)(next.first.first<<16)+(next.first.second)}));
        }
        cout << endl;
    }

    pair<int, pair<int, int> > ret_bit;
    for(int i = 0; i<latice->numInputs;i++)
        for(int j = 0;j<latice->numInputs;j++)
        {
            //cout << votes[i] <<" ";
            ret_bit = max(ret_bit, make_pair(votes[i][j], make_pair(i, j)));
        }
    //cout << endl;

    return make_pair(make_pair(latice->circuit[ret_bit.second.first], latice->circuit[ret_bit.second.second]), ret_bit.first);
}

pair<pair<bit_signature, bit_signature>, int>
ensamble_pair_teacher
        (
                Data* latice,
                int num_teachers,
                NeuralNetwork::parameters param,
                NeuralNetwork::PriorityTrainReport (NeuralNetwork::*training_f)(Data*, NeuralNetwork::parameters)
        )
{
    vector<NeuralNetwork::data_model::bit_dimension_pair> empty;
    return ensamble_pair_teacher(latice, num_teachers, param, training_f, false, empty);
}