//
// Created by Kliment Serafimov on 2020-01-02.
//

#include "dataset_initializers.h"
#include "DecisionTreeSynthesisViaDP.h"


FirstOrderDataset<DataAndDecisionTreeScore> init_random_test_set(int n)
{
    FirstOrderDataset<DataAndDecisionTreeScore> ret;
    int test_num_samples = 64;

    vector<FunctionAndDecisionTreeScore> sample_ids;

    ///VVVVVV THIS

    //WHAT IS THE SMALLES TRAINING SET OF ({0, 1}^4 -> {0, 1}) -> {2x+1 | 0 <= x <= 8}
    //s.t opt decision tree on function {0, 1}^4 -> {0, 1} generalizes to total opt.
    //What types of function? LET U_2^2^2^2 = {0, 1}^4 -> {0, 1}
    //{f-> 1_[if (DT_opt(f) == x)] where \forall f \in U_2^2^2^2  }_x \in {2x+1 | 0 <= x <= 8}
    //{f-> 1_[if (DT_opt(f) <= x)] where \forall f \in U_2^2^2^2  }_x \in {2x+1 | 0 <= x <= 8}
    //{f-> 1_[if (((DT_opt(f)-1) >> x)%2 == 0)] where \forall f \in U_2^2^2^2  }_x \in {1, 2, 3}
    //U_2^2^U_2^2^2^2 = {{0, 1}^4 -> {0, 1}} -> {0, 1}

    //Model the curriculum with functions from U_2^2^U_2^2^2^2
    //for train curriculum one tree tree
    //for test curriculum another tree
    // .. other test curriculums
    //Describe model that goes from one curriculum to another.
    //Search over this space with NN initializations that generate according optimal f\in U_2^2^2^2 DT_neural_error = DT_opt
    //without training on all of them.

    //Amphibian synthesis


    //(2^2^16) functions with 16 bit input
    //

    ///^^^^^^^^^^^^THIS Came from the fact that


    for(int i = 0; i < test_num_samples;i++)
    {
        long long first = (long long)rand(0, (1<<31)) << 31;
        long long second = rand(0, (1 << 31));

        long long sample_id = (first + second);

        //cout << first << " " << second << " "<< sample_id << endl;
        sample_ids.push_back(FunctionAndDecisionTreeScore(sample_id, get_opt_decision_tree_score(n, sample_id)));
    }

    //test

    for(int i = 0;i<test_num_samples;i++)
    {
        DataAndDecisionTreeScore new_data(sample_ids[i]);
        new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].function);
        ret.test_data.push_back(new_data);
    }

    cout << ret.print(0) << endl;

    return ret;
}

FirstOrderDataset<DataAndDecisionTreeScore> init_custom_data_and_difficulty_set(int n, double training_set_percent, double testing_set_percent)
{
    assert(n<=4);
    FirstOrderDataset<DataAndDecisionTreeScore> ret;

    //int train_num_samples; //defined later
    int test_num_samples = (1<<(1<<n));

    vector<FunctionAndDecisionTreeScore> sample_ids = get_smallest_f(n);

    if(n <= 4)
    {
        for(int i = 0;i<sample_ids.size();i++)
        {
            DataAndDecisionTreeScore new_data(sample_ids[i]);
            new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].function);

            ret.test_data.push_back(new_data);
            ret.train_data.push_back(new_data);
//            if (rand(0, 100) < testing_set_percent) {
//                ret.test_data.push_back(new_data);
//            } else {
//                if (rand(0, 100) < training_set_percent) {
//                    ret.train_data.push_back(new_data);
//                }
//            }
        }
    }
    else {

        //truly custom
        assert(0);

        if (n == 3) {

            srand(0);

            for (int i = 0; i < sample_ids.size(); i++) {
                if (sample_ids[i].size < 7) {
                    DataAndDecisionTreeScore new_data(sample_ids[i]);
                    new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].function);

                    if (rand(0, 100) < testing_set_percent) {
                        ret.test_data.push_back(new_data);
                    } else {
                        if (rand(0, 100) < training_set_percent) {
                            ret.train_data.push_back(new_data);
                        }
                    }
                }
            }


            if (true) {
                int original_seed = 0;
                for (int i = 0; i < 8; i++) {
                    int seed = original_seed;
                    if ((seed & (1 << i)) != 0) {
                        seed -= (1 << i);
                    } else {
                        seed |= (1 << i);
                    }

                    DataAndDecisionTreeScore new_1_data(sample_ids[seed]);
                    {
                        new_1_data.init_exaustive_table_with_unary_output(n, sample_ids[seed].function);

                        if (false) {
                            if (rand(0, 100) < testing_set_percent) {
                                ret.test_data.push_back(new_1_data);
                            } else {
                                if (rand(0, 100) < training_set_percent) {
                                    ret.train_data.push_back(new_1_data);
                                }
                            }
                        }
                    }
                    int local_seed = seed;
                    for (int j = i + 1; j < 8; j++) {
                        seed = local_seed;
                        if ((seed & (1 << j)) != 0) {
                            seed -= (1 << j);
                        } else {
                            seed |= (1 << j);
                        }
                        DataAndDecisionTreeScore new_2_data(sample_ids[seed]);
                        if (new_2_data.score.size >= 9) {
                            new_2_data.init_exaustive_table_with_unary_output(n, sample_ids[seed].function);

                            if (false) {
                                if (rand(0, 100) < testing_set_percent) {
                                    ret.test_data.push_back(new_2_data);
                                } else {
                                    if (rand(0, 100) < training_set_percent) {
                                        ret.train_data.push_back(new_2_data);
                                    }
                                }
                            }
                        }
                        int local_local_seed = seed;
                        for (int k = j + 1; k < 8; k++) {
                            seed = local_local_seed;
                            if ((seed & (1 << k)) != 0) {
                                seed -= (1 << k);
                            } else {
                                seed |= (1 << k);
                            }
                            DataAndDecisionTreeScore new_3_data(sample_ids[seed]);
                            if (sample_ids[seed].size == 7) {
                                if (true) {
                                    new_3_data.init_exaustive_table_with_unary_output(n, sample_ids[seed].function);

                                    if (rand(0, 100) < testing_set_percent) {
                                        ret.test_data.push_back(new_3_data);
                                    } else {
                                        if (rand(0, 100) < training_set_percent) {
                                            ret.train_data.push_back(new_3_data);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        //train
        if (n <= 2) {

            for (int i = 0; i < sample_ids.size(); i++) {
                int size = sample_ids[i].size;
                if
                        (n <= 2 ||
                         (
                                 n == 3 && (size <= 3
                                            || (size == 5 && rand(0, 100) < 20) || (size == 7 && rand(0, 100) < 7)
                                            || (size == 9 && rand(0, 100) < 7) || (size == 11 && rand(0, 100) < 16)
                                            || (size == 13 && rand(0, 100) < 30) || (size == 15))
                         ) ||
                         (n == 4 && (size <= 12))
                        )// || size >= 15 || rand(0, 256) < 24)))
                {
                    DataAndDecisionTreeScore new_data(sample_ids[i]);
                    new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].function);
                    ret.train_data.push_back(new_data);
                }

                /*
                if
                        (n <= 2 ||
                         (
                                 n == 3 && (size <= 3
                                            || (size == 5 && rand(0, 100) < 3400) || (size == 7 && rand(0, 100) < 10)
                                            || (size == 9 && rand(0, 100) < 10) || (size == 11 && rand(0, 100) < 10)
                                            || (size == 13 && rand(0, 100) < 20) || (size == 15))
                         ) ||
                         (n == 4 && (size <= 12))
                        )// || size >= 15 || rand(0, 256) < 24)))
                {
                    DataAndDecisionTreeScore new_data(sample_ids[i]);
                    new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].f);
                    ret.test_data.push_back(new_data);
                }
                */
            }
            sample_ids.clear();

            sample_ids = get_smallest_f(n);

            //test

            for (int i = 0; i < test_num_samples; i++) {
                DataAndDecisionTreeScore new_data(sample_ids[i]);
                new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].function);
                ret.test_data.push_back(new_data);
            }
        }
    }


    cout << ret.print(0) << endl;

    return ret;
}

/*
FirstOrderDataset<DataAndDecisionTreeScore> init_smallest_train_set(int n)
{
    FirstOrderDataset<DataAndDecisionTreeScore> ret;
    ret.n = n;
    //train
    vector<int> function_ids ;

    int train_num_samples = 256;
    int test_num_samples = (1<<(1<<n));

    vector<FunctionAndDecisionTreeScore> sample_ids = get_smallest_f(n, train_num_samples);

    //sample_ids.push_back(0);
    //sample_ids.push_back((1<<(1<<n))-1);

    //train
    for (int i = 0; i < train_num_samples; i++)
    {
        DataAndDecisionTreeScore new_data(sample_ids[i].size, sample_ids[i].num_solutions);
        new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].f);
        ret.train_data.push_back(new_data);
    }

    sample_ids.clear();

    if(n<=4)
    {
        sample_ids = get_smallest_f(n, test_num_samples);
    }
    else if(n<=6)
    {
        test_num_samples = 200;
        for(int i = 0; i < max(train_num_samples, test_num_samples);i++)
        {
            long long first = (long long)rand(0, (1<<31)) << 31;
            long long second = rand(0, (1 << 31));

            long long sample_id = (first + second);

            //cout << first << " " << second << " "<< sample_id << endl;
            sample_ids.push_back(FunctionAndDecisionTreeScore(sample_id, get_opt_decision_tree_score(n, sample_id)));
        }
    }
    else
    {
        assert(0);
    }

    //test

    for(int i = 0;i<test_num_samples;i++)
    {
        DataAndDecisionTreeScore new_data(sample_ids[i].size, sample_ids[i].num_solutions);
        new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].f);
        ret.test_data.push_back(new_data);
    }

    ret.print();

    return ret;
}
*/

void init_exaustive_table_with_unary_output(Data &f_data, int n, int f_id)
{
    f_data.init_exaustive_table_with_unary_output(n, f_id);
}

void init_exaustive_table_with_unary_output(DataAndDecisionTreeScore &f_data, int n, int f_id)
{
    f_data.init_exaustive_table_with_unary_output(n, f_id);

    DecisionTreeSynthesisViaDP<Data> decision_tree_solver;
    NeuralNetwork::parameters cutoff_parametes;
    DecisionTreeScore size_opt = decision_tree_solver.synthesize_decision_tree_and_get_size
            (cutoff_parametes, n, f_data, optimal);

    f_data.score = size_opt;
}

vector<vector<Data> > initSecondOrderDataset(vector<Data> data, int subDataset_size)
{
    vector<vector<Data> > ret;
    for(int i = 0, meta_data_id = 0;i<data.size();meta_data_id++) {
        ret.push_back(vector<Data>());
        for (int k = 0; k < subDataset_size && i < data.size(); i++, k++) {
            ret[meta_data_id].push_back(data[i]);
        }
    }
    return ret;
}

SecondOrderDataset<Data> initSecondOrderDatasets(FirstOrderDataset<Data> data)
{
    SecondOrderDataset<Data> ret;
    int subDataset_size = 2;
    ret.train_meta_data = initSecondOrderDataset(data.train_data, subDataset_size);
    ret.test_meta_data = initSecondOrderDataset(data.test_data, subDataset_size);

    return ret;
}

