//
// Created by Kliment Serafimov on 2020-01-05.
//

#include "LatentDecisionTreeExtractor.h"

vector<vector<pair<pair<double, double>, DecisionTreeScore> > > get_ensamble_neural_error_buckets
        (int n, FirstOrderDataset<DataAndDecisionTreeScore> first_order_data, NeuralNetwork::parameters params)
{
    vector<vector<pair<pair<double, double>, DecisionTreeScore> > > ret;

    vector<DataAndDecisionTreeScore> sorted;

    vector<NeuralNetworkAndScore*> ensamble_nets = params.progressive_ensamble_nets[n];

    int leaf_iter = params.get_iteration_count(n);

    sorted = first_order_data.train_data;

    sort_v(sorted);

    first_order_data.train_data = sorted;


    for(int i = 0; i < ensamble_nets.size(); i++)
    {
        cout << "HERE " << ensamble_nets[i]->print() << endl;
        vector<double> sums;
        vector<int> count;
        bool prev_score_defined = false;
        DecisionTreeScore prev_score;

        //cout << "size = " << first_order_data.train_data.size() <<endl;


        vector<pair<pair<double, double>, DecisionTreeScore> > overlapping;

        cout << "first_order_data.train_data.size() = " << first_order_data.train_data.size() <<endl;

        for(int j = 0; j < first_order_data.train_data.size();j++)
        {


            NeuralNetwork leaf_learner = ensamble_nets[i];
//            NeuralNetwork::parameters leaf_parameters = cutoff_param(leaf_iter, 0.01);
            params.set_iteration_count(leaf_iter);
            leaf_learner.train(&first_order_data.train_data[j], params, params.priority_train_f);
            double error = leaf_learner.get_error_of_data(&first_order_data.train_data[j]).second;

            cout << first_order_data.train_data[j].printConcatinateOutput() << "\t" << first_order_data.train_data[j].score.size << "\t" << error << endl;

            if(prev_score_defined)
            {
                if(prev_score.size == first_order_data.train_data[j].score.size)
                {
                    int last_id = sums.size()-1;
                    overlapping[last_id].first.first = min(error, overlapping[last_id].first.first);
                    overlapping[last_id].first.second = max(error, overlapping[last_id].first.second);
                    sums[last_id] += error;
                    count[last_id] += 1;
                }
                else
                {
                    overlapping.push_back(make_pair(make_pair(error, error), first_order_data.train_data[j].score));
                    sums.push_back(error);
                    count.push_back(1);
                    prev_score = first_order_data.train_data[j].score;
                }
            }
            else
            {
                overlapping.push_back(make_pair(make_pair(error, error), first_order_data.train_data[j].score));
                sums.push_back(error);
                count.push_back(1);
                prev_score = first_order_data.train_data[j].score;
                prev_score_defined = true;
            }

        }

        assert(sums.size() == count.size());

        cout <<"n = " << n << " sums.size() == " << sums.size() <<endl;


        for(int local_i = 0;local_i<(int)sums.size()-1;local_i++)
        {
            assert(local_i<sums.size()-1);
//            cout << local_i << " "<< sums.size()-1 << endl;
            cout << "local_i = " << local_i <<endl;
            cout << "sums[local_i] = " << sums[local_i]<<" | count[local_i] = " << count[local_i] <<" | ";
            double at_now = sums[local_i]/count[local_i];
            double at_next = sums[local_i]/count[local_i];
            double mid = (at_now+at_next)/2;
            if(overlapping[local_i].first.second < overlapping[local_i+1].first.first)
            {
                //all good
            }
            else
            {
                overlapping[local_i].first.second = mid;
                overlapping[local_i + 1].first.first = mid;
            }
            cout << "at_now = " << at_now <<" | " << "at_next = " << at_next <<" | mid = " << mid << endl;
        }

        overlapping[sums.size()-1].first.second = 1;

        ret.push_back(overlapping);

    }

    cout << "MODEL BRACKETS:" <<endl;

    for(int i = 0;i<ret.size();i++)
    {
        for(int j = 0;j<ret[i].size();j++)

        {
            cout << ret[i][j].first.first <<" "<< ret[i][j].first.second <<" "<< ret[i][j].second.size <<" | " <<endl;
            assert(ret[i][j].first.first <= ret[i][j].first.second);
            if(j>=1)
            {
                assert(ret[i][j-1].first.second <= ret[i][j].first.first);
            }
        }
    }


    return ret;
}


vector<int> hidden_layer_block(int width, int depth)
{
    vector<int> ret;
    for(int i = 0;i<depth;i++)
    {
        ret.push_back(width);
    }
    return ret;
}


vector<vector<vector<pair<pair<double, double>, DecisionTreeScore> > > > get_progressive_ensamble_neural_error_buckets
        (int n, vector<FirstOrderDataset<DataAndDecisionTreeScore> > first_order_datasets, NeuralNetwork::parameters params)
{

    vector<vector<vector<pair<pair<double, double>, DecisionTreeScore> > > > ret;
    ret.push_back(vector<vector<pair<pair<double, double>, DecisionTreeScore> > > ());
    for(int i = 1; i <= n; i++)
    {
        if(first_order_datasets[i].train_data.size()>=1) {
            ret.push_back(get_ensamble_neural_error_buckets
                                  (i, first_order_datasets[i], params));
        } else{
            ret.push_back(vector<vector<pair<pair<double, double>, DecisionTreeScore> > >());
        }
    }
    return ret;
}