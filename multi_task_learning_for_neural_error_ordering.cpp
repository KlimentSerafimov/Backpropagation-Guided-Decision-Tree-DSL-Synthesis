//
// Created by Kliment Serafimov on 2020-01-05.
//

#include "multi_task_learning_for_neural_error_ordering.h"
#include "LanguagesOverBooleanFunctions.h"

vector<DataAndDecisionTreeScore> select_all_functions(int n)
{
    assert(n<=3);

    //custom languages
    if(true) {
        LanguagesOverBooleanFunctions language = LanguagesOverBooleanFunctions(n, 10);
        language.enumerate();

        vector<DataAndDecisionTreeScore> ret;

        for(int i  = 0;i<language.ordering_over_boolean_functions.size();i++)
        {
            int f_as_int = (int) language.ordering_over_boolean_functions[i].total_function.to_ullong();
            DataAndDecisionTreeScore new_data(FunctionAndDecisionTreeScore(f_as_int, i >= language.bucket_0));
            new_data.init_exaustive_table_with_unary_output(n, f_as_int);
            ret.push_back(new_data);
        }
        return ret;
    }
    else {
        //decision trees
        vector<DataAndDecisionTreeScore> ret;

        vector<FunctionAndDecisionTreeScore> sample_ids = get_smallest_f(n);

        if (n == 3) {
            srand(0);
            for (int i = 0; i < sample_ids.size(); i++) {
                if (sample_ids[i].size < 9) {
                    DataAndDecisionTreeScore new_data(sample_ids[i]);
                    new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].function);

                    ret.push_back(new_data);
                } else if (false && /*sample_ids[i].size <= 9 && */__builtin_popcount(i) <= 2) {
                    DataAndDecisionTreeScore new_data(sample_ids[i]);
                    new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].function);

                    ret.push_back(new_data);
                }
            }
        } else {
            assert(0);
        }

        return ret;
    }
}

FirstOrderDataset<DataAndDecisionTreeScore> split_into_first_order_dataset(
        vector<DataAndDecisionTreeScore> all_functions, double training_set_size)
{
    FirstOrderDataset<DataAndDecisionTreeScore> ret;

    for(int i = 0, remaining = (int) all_functions.size(); i<all_functions.size(); i++, remaining--)
    {
        if(
                all_functions.size()*training_set_size - ret.train_data.size() < remaining &&
                (all_functions.size()-all_functions.size()*training_set_size) - ret.test_data.size() < remaining) {
            if (rand(0, 100) < training_set_size * 100) {
                ret.train_data.push_back(all_functions[i]);
            } else {
                ret.test_data.push_back(all_functions[i]);
            }
        }
        else if(all_functions.size()*training_set_size - ret.train_data.size() >= remaining ) {

            ret.train_data.push_back(all_functions[i]);
        }
        else if((all_functions.size()-all_functions.size()*training_set_size) - ret.test_data.size() >= remaining){
            ret.test_data.push_back(all_functions[i]);
        }
        else
        {
            assert(0);
        }
    }

    cout << "INIT all_functions.size() = " << all_functions.size() << endl;
    cout << "INIT training_set_size = " << training_set_size <<endl;
    cout << "ret.train_data.size() = " << ret.train_data.size() << endl;
    cout << "ret.test_data.size() = " << ret.test_data.size() << endl;


    return ret;
}


vector<DataAndDecisionTreeScore> take_percentage(vector<DataAndDecisionTreeScore> all_data, double take_percent)
{
    vector<DataAndDecisionTreeScore> ret;
    for(int i = 0;i<all_data.size();i++)
    {
        if(rand(0, 100) <= take_percent*100)
        {
            ret.push_back(all_data[i]);
        }
    }

    return ret;
}


vector<FirstOrderDataset<DataAndDecisionTreeScore> > sample_partitions(vector<DataAndDecisionTreeScore> all_data, int num_samples)
{
    vector<FirstOrderDataset<DataAndDecisionTreeScore> > ret;

    for(int i = 0;i<num_samples;i++)
    {
        ret.push_back(split_into_first_order_dataset(all_data, 0.5));
    }

    return ret;
}

void multi_task_learning_for_neural_error_ordering()
{
    int local_n = 3;

    vector<DataAndDecisionTreeScore> all_functions = select_all_functions(local_n);

    srand(0);
    FirstOrderDataset<DataAndDecisionTreeScore> first_order_dataset =
            split_into_first_order_dataset(all_functions, 0.5);

    //FirstOrderDataset<FirstOrderDataset<DataType> >
    generalizationDataset<DataAndDecisionTreeScore> holy_grail;

    int num_samples = 3;
    holy_grail.test_data = sample_partitions(first_order_dataset.test_data, num_samples);

    int num_repeats = 1;

    for (double percent_of_training_set = 1; percent_of_training_set <= 1; percent_of_training_set += 0.3)
    {

        vector<DataAndDecisionTreeScore> sub_training_set = take_percentage(
                first_order_dataset.train_data, percent_of_training_set);

        holy_grail.train_data = sample_partitions(sub_training_set, num_samples);

        cout << holy_grail.print(0) << endl;

        for(int num_amphibian_iter = 10; num_amphibian_iter <= 1000 ; num_amphibian_iter+=10)
        {

            for(int repeat = 0; repeat < num_repeats; repeat++) {
                srand(repeat);
                NeuralNetworkAndScore at_init =  NeuralNetworkAndScore(NeuralNetwork(local_n, 2 * local_n, 1));
                //param definition;
                FirstOrderLearning<DataAndDecisionTreeScore>::learning_parameters param;
                param.at_initiation_of_parameters_may_21st_10_52_pm();
                param.choosePriorityTrain = &NeuralNetwork::stocasticPriorityTrain;

                param.leaf_iter_init = 60;
                param.max_boundary_size = 3;
                param.max_num_iter = 20;


                int sum = 0;

                for (int i = 0; i < holy_grail.test_data.size(); i++) {
                    cout << "-----------------------------------------------------------------" << endl;
                    cout << "NOW PRE-TESTING ON FIRST_ORDER_DATASET.TEST[" << i << "]" << endl;
                    NeuralNetworkAndScore result = improve_initialization<DataAndDecisionTreeScore>(
                            local_n, at_init, holy_grail.test_data[i], param, true);
                    sum += result.sum_ordering_error;
                }

                double pre_train_test_erorr = (double) sum / holy_grail.test_data.size();
                cout << "pre_train_test_error = " << pre_train_test_erorr << endl;


                for (int i = 0; i < num_amphibian_iter; i++) {
                    int first_order_dataset_id = rand(0, holy_grail.train_data.size() - 1);

                    cout << "-----------------------------------------------------------------" << endl;
                    cout << "ITER = " << std::to_string(i) << "; NOW TRAINING ON FIRST_ORDER_DATASET.TRAIN[" << first_order_dataset_id << "]" << endl;

                    at_init = improve_initialization<DataAndDecisionTreeScore>(
                            local_n, at_init, holy_grail.train_data[first_order_dataset_id], param, false);
                }


                sum = 0;

                for (int i = 0; i < holy_grail.train_data.size(); i++) {
                    cout << "-----------------------------------------------------------------" << endl;
                    cout << "NOW TESTING ON FIRST_ORDER_DATASET.TRAIN[" << i << "]" << endl;
                    NeuralNetworkAndScore result = improve_initialization<DataAndDecisionTreeScore>(local_n, at_init,
                                                                                                    holy_grail.train_data[i], param, false);
                    sum += result.sum_ordering_error;
                }

                double post_train_train_error = (double) sum / holy_grail.train_data.size();
                cout << "post_train_train_error = " << post_train_train_error << endl;

                sum = 0;

                for (int i = 0; i < holy_grail.test_data.size(); i++) {
                    cout << "-----------------------------------------------------------------" << endl;
                    cout << "NOW TESTING ON FIRST_ORDER_DATASET.TEST[" << i << "]" << endl;
                    NeuralNetworkAndScore result = improve_initialization<DataAndDecisionTreeScore>(
                            local_n, at_init, holy_grail.test_data[i], param, true);
                    sum += result.sum_ordering_error;
                }

                double post_train_test_error = (double) sum / holy_grail.test_data.size();
                cout << "post_train_test_error = " << post_train_test_error << endl;

                fout_experiment << num_amphibian_iter << "\t" << sub_training_set.size() << "\t"
                                << pre_train_test_erorr << "\t" << post_train_test_error << "\t"
                                << post_train_train_error << endl;
            }
        }
    }
}