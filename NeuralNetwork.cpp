//
// Created by Kliment Serafimov on 2020-01-02.
//

#include "NeuralNetwork.h"

NeuralNetwork::parameters choose_learning_rate_standard_param(double treshold, double rate)
{
    NeuralNetwork::parameters param = NeuralNetwork::parameters(rate, 1);
    param.accuracy = treshold;
    param.track_dimension_model = false;
    param.priority_train_f = &NeuralNetwork::softPriorityTrain;
    return param;
}

NeuralNetwork::parameters choose_learning_rate_cutoff_param(int cutoff_iter, double treshold, double rate)
{
    //cout << "rate = " << rate << ", ";
    NeuralNetwork::parameters param = choose_learning_rate_standard_param(treshold, rate);
    param.set_iteration_count(cutoff_iter);
    return param;
}

NeuralNetwork::parameters standard_param(double treshold)
{
    NeuralNetwork::parameters param = NeuralNetwork::parameters(1, 1);
    param.accuracy = treshold;
    param.track_dimension_model = false;
    param.priority_train_f = &NeuralNetwork::softPriorityTrain;
    return param;
}

NeuralNetwork::parameters cutoff_param(int cutoff_iter, double treshold)
{
    NeuralNetwork::parameters param = standard_param(treshold);
    param.set_iteration_count(cutoff_iter);
    return param;
}

NeuralNetwork::parameters meta_cutoff(int root_iter, int leaf_iter)
{
    NeuralNetwork::parameters param = standard_param(0.01);
    param.set_meta_iteration_count(root_iter);
    param.set_iteration_count(leaf_iter);
    return param;
}