//
// Created by Kliment Serafimov on 2020-01-02.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_BITVECTORFUNCTIONSOLVER_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_BITVECTORFUNCTIONSOLVER_H

#include "SecondOrderLearning.h"
#include "FirstOrderLearning.h"
#include "FirstOrderDataset.h"
#include "SecondOrderDataset.h"

template<typename DataType>
class BitvectorFunctionSolver: public SecondOrderLearning<DataType>
{
    typedef SecondOrderLearning<DataType> base;
public:
    int root_iter = 60, leaf_iter = 60;

    BitvectorFunctionSolver(int _leaf_iter): SecondOrderLearning<DataType>()
    {
        assert(_leaf_iter >= 1);
        leaf_iter = _leaf_iter;
    }

    FirstOrderDataset<DataType> first_order_data;
    SecondOrderDataset<DataType> meta_data;

    bool first_order_data_inited = false;
    bool meta_data_inited = false;

    void initFirstOrderData(int n, FirstOrderDataset<DataType> _first_order_data)
    {
        first_order_data = _first_order_data;
        first_order_data_inited = true;
    }

    void initMetaData(int n, FirstOrderDataset<DataType> _first_order_data)
    {

        first_order_data = _first_order_data;
        first_order_data_inited = true;

        assert(first_order_data.train_data.size() >= 1);
        assert(first_order_data.test_data.size() >= 1);

        meta_data = initSecondOrderDatasets(first_order_data);

        assert(meta_data.train_meta_data.size() >= 1);
        assert(meta_data.test_meta_data.size() >= 1);

        meta_data_inited = true;
    }

    const double treshold = 0.4;
    double min_treshold = treshold/3;

    NeuralNetworkAndScore train_to_order_neural_errors
            (int n, NeuralNetworkAndScore learner, typename FirstOrderLearning<DataType>::learning_parameters param, bool print)
    {
        assert(first_order_data_inited);

        base::order_neural_errors(learner, first_order_data.train_data, param, print);

        return learner;
    }

    NeuralNetworkAndScore train_to_learn(int n, typename FirstOrderLearning<DataType>::learning_parameters param, bool print)
    {
        assert(first_order_data_inited);

        srand(time(0));
        NeuralNetworkAndScore learner = NeuralNetworkAndScore(NeuralNetwork(n, 2*n, 1));

        return train_to_learn(n, learner, param, print);
    }

    NeuralNetworkAndScore train_to_learn(int n, NeuralNetworkAndScore learner, typename FirstOrderLearning<DataType>::learning_parameters param, bool print)
    {
        assert(first_order_data_inited);

        base::meta_learn(learner, first_order_data.train_data, param, print);

        min_treshold = learner.max_error;
        leaf_iter = learner.max_leaf_iter;

        assert(leaf_iter != 0);

        return learner;
    }


    void test_to_learn(NeuralNetworkAndScore learner)
    {
        assert(first_order_data_inited);

        meta_net_and_score rez =
                evaluate_learner(learner, first_order_data.test_data, true, leaf_iter, min_treshold);

        learner.printWeights();
        cout << "rez = " << rez.print() <<endl;
    }

    meta_net_and_score train_to_meta_learn(int n)
    {
        assert(meta_data_inited);

        srand(time(0));
        meta_net_and_score learner = NeuralNetworkAndScore(NeuralNetwork(n, 2*n, 1));

        return train_to_meta_learn(n, learner);
    }

    meta_net_and_score train_to_meta_learn(int n, meta_net_and_score learner)
    {
        assert(meta_data_inited);

        typename FirstOrderLearning<DataType>::evaluate_learner_parameters param;
        param.iter_cutoff = leaf_iter;
        param.threshold = treshold;

        assert(meta_data.train_meta_data.size() >= 1);
        base::learn_to_meta_learn(learner, meta_data.train_meta_data, root_iter, param);

        min_treshold = learner.max_error;
        leaf_iter = learner.max_leaf_iter;

        assert(leaf_iter != 0);

        return learner;
    }

    void test_to_meta_learn(meta_net_and_score learner)
    {
        assert(meta_data_inited);

        typename FirstOrderLearning<DataType>::evaluate_learner_parameters params;
        params.iter_cutoff = leaf_iter;
        params.threshold = min_treshold;

        meta_net_and_score rez =
                base::evaluate_meta_learner(learner, meta_data.test_meta_data, true, root_iter, params);

        learner.printWeights();
        cout << "rez = " << rez.print() <<endl;
    }

};


#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_BITVECTORFUNCTIONSOLVER_H
