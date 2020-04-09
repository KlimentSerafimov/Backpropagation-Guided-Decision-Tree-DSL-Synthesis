//
// Created by Kliment Serafimov on 2020-01-05.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_MULTI_TASK_LEARNING_FOR_NEURAL_ERROR_ORDERING_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_MULTI_TASK_LEARNING_FOR_NEURAL_ERROR_ORDERING_H

#include "NeuralNetworkAndScore.h"
#include "FirstOrderDataset.h"
#include "DataAndDecisionTreeScore.h"
#include "FunctionAndDecisionTreeScore.h"
#include "FirstOrderLearning.h"
#include "DecisionTreeSynthesisViaDP.h"

static ofstream fout_experiment("experiment1.out");

template<typename DataType>
NeuralNetworkAndScore improve_initialization(
        int local_n,
        NeuralNetworkAndScore init_initialization,
        FirstOrderDataset<DataType> first_order_dataset,
        typename FirstOrderLearning<DataType>::learning_parameters param,
        bool print
)
{

    FirstOrderLearning<DataType> trainer;

    NeuralNetworkAndScore improved_initialization  =
            trainer.order_neural_errors
                    (init_initialization, first_order_dataset.train_data, param, false, false);

    NeuralNetworkAndScore test_good_initialization =
            trainer.evaluate_learner
                    (improved_initialization, first_order_dataset.test_data, false, param, &NeuralNetwork::softPriorityTrain);

    Policy test_policy = Policy(first_order_dataset.test_data);
    test_policy.update(&test_good_initialization);

    if(print)
    {
        trainer.print_ordering(test_policy, first_order_dataset.test_data, test_good_initialization.tarjans);

        test_good_initialization.printWeights();
    }

    return test_good_initialization;
}

template<typename DataType>
class generalizationDataset: public FirstOrderDataset<FirstOrderDataset<DataType> >
{
};

vector<DataAndDecisionTreeScore> select_all_functions(int n);

FirstOrderDataset<DataAndDecisionTreeScore> split_into_first_order_dataset(
        vector<DataAndDecisionTreeScore> all_functions, double training_set_size);

vector<DataAndDecisionTreeScore> take_percentage(vector<DataAndDecisionTreeScore> all_data, double take_percent);

vector<FirstOrderDataset<DataAndDecisionTreeScore> > sample_partitions(vector<DataAndDecisionTreeScore> all_data, int num_samples);

void multi_task_learning_for_neural_error_ordering();


#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_MULTI_TASK_LEARNING_FOR_NEURAL_ERROR_ORDERING_H
