//
// Created by Kliment Serafimov on 2020-01-02.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DATASET_INITIALIZERS_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DATASET_INITIALIZERS_H

#include "DataAndDecisionTreeScore.h"
#include "Data.h"
#include "SecondOrderDataset.h"
#include "FirstOrderDataset.h"
#include "FunctionAndDecisionTreeScore.h"

FirstOrderDataset<DataAndDecisionTreeScore> init_random_test_set(int n);

FirstOrderDataset<DataAndDecisionTreeScore> init_custom_data_and_difficulty_set(int n, double training_set_percent, double testing_set_percent);

/*
FirstOrderDataset<DataAndDecisionTreeScore> init_smallest_train_set(int n);
*/

void init_exaustive_table_with_unary_output(Data &f_data, int n, int f_id);

void init_exaustive_table_with_unary_output(DataAndDecisionTreeScore &f_data, int n, int f_id);

template<typename DataType>
FirstOrderDataset<DataType> initFirstOrderDatasets(int n, int train_num_samples, int test_num_samples)
{
    FirstOrderDataset<DataType> ret;

    vector<long long> sample_ids;

    //common
    if(n<=4) {

        for (int i = 0; i < (1 << (1 << n)); i++) {
            sample_ids.push_back(i);
        }

        //shuffle(sample_ids.begin(), sample_ids.end(), std::default_random_engine(0));

    }
    else if(n<=6)
    {
        for(int i = 0; i < test_num_samples;i++)
        {
            long long first = (long long)rand(0, (1<<31)) << 31;
            long long second = rand(0, (1 << 31));

            long long sample_id = (first + second);

            //cout << first << " " << second << " "<< sample_id << endl;
            sample_ids.push_back(sample_id);
        }
    }
    else
    {
        assert(0);
    }

    //train
    for (int i = 0; i < train_num_samples; i++) {

        DataType next_f;

        init_exaustive_table_with_unary_output(next_f, n, sample_ids[i]);

        ret.train_data.push_back(next_f);
    }

    //test

    for(int i = 0;i<test_num_samples;i++)
    {
        DataType next_f;

        init_exaustive_table_with_unary_output(next_f, n, sample_ids[i]);

        ret.test_data.push_back(next_f);
    }

    return ret;
}

template<typename DataType>
FirstOrderDataset<DataType> completeFirstOrderTestDataset(int n)
{
    return initFirstOrderDatasets<DataType>(n, (1<<(1<<n)), (1<<(1<<n)));
}

vector<vector<Data> > initSecondOrderDataset(vector<Data> data, int subDataset_size);

SecondOrderDataset<Data> initSecondOrderDatasets(FirstOrderDataset<Data> data);


#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DATASET_INITIALIZERS_H
