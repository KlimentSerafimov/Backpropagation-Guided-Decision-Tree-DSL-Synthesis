//
// Created by Kliment Serafimov on 2020-01-05.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_LATENTDECISIONTREEEXTRACTOR_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_LATENTDECISIONTREEEXTRACTOR_H

#include <vector>
#include "DecisionTreeScore.h"
#include "DataAndDecisionTreeScore.h"
#include "NeuralNetworkAndScore.h"
#include "FirstOrderDataset.h"
#include "FirstOrderLearning.h"
#include "SecondOrderLearning.h"
#include "BitvectorFunctionSolver.h"
#include "DecisionTreeSynthesisViaDP.h"
#include "dataset_initializers.h"


static ofstream fout("table.out");

using namespace std;

vector<vector<pair<pair<double, double>, DecisionTreeScore> > > get_ensamble_neural_error_buckets
        (int n, FirstOrderDataset<DataAndDecisionTreeScore> first_order_data, NeuralNetwork::parameters params);

vector<int> hidden_layer_block(int width, int depth);

vector<vector<vector<pair<pair<double, double>, DecisionTreeScore> > > > get_progressive_ensamble_neural_error_buckets
        (int n, vector<FirstOrderDataset<DataAndDecisionTreeScore> > first_order_datasets, NeuralNetwork::parameters params);

template<typename DataType>
vector<NeuralNetworkAndScore*> get_local_ensamble(
        int local_n,
        vector<int> hidden_layer_width,
        vector<int> hidden_layer_depth,
        int num_ensambles,
        vector<FirstOrderDataset<DataType> > first_order_datasets,
        NeuralNetworkAndScore ensamble_progressive_nets[10][10],
        typename FirstOrderLearning<DataType>::learning_parameters all_params)
{
    vector<NeuralNetworkAndScore*> local_ensamble;

    for(int ensamble_id = 0; ensamble_id < num_ensambles; ensamble_id++)
    {
        vector<int> hidden_layer_widths = hidden_layer_block(hidden_layer_width[local_n], hidden_layer_depth[local_n]);

        NeuralNetworkAndScore radnom_init = NeuralNetworkAndScore(NeuralNetwork(local_n, hidden_layer_widths, 1));

        NeuralNetworkAndScore improved_initialization = improve_initialization(local_n, radnom_init, first_order_datasets[local_n], all_params, true);

        ensamble_progressive_nets[local_n][ensamble_id] = improved_initialization;

        local_ensamble.push_back(&ensamble_progressive_nets[local_n][ensamble_id]);
    }

    return local_ensamble;
}

class LatentDecisionTreeExtractor: public SecondOrderLearning<Data>
{
public:

    LatentDecisionTreeExtractor(): SecondOrderLearning()
    {
    }

    class SynhtesizerScoreCoparitor
    {
    public:
        int sum_distance = 0;
        double sum_ratio = 0;
        double max_ratio = 0;
        int max_distance = 0;
        int count = 0;

        void clear_score()
        {
            sum_distance = 0;
            sum_ratio = 0;

            max_ratio = 0;
            max_distance = 0;
            count = 0;
        }

        void compose_with(DecisionTreeScore _base, DecisionTreeScore _other)
        {
            int base = _base.size;
            int other = _other.size;

            int distance = other - base;
            double ratio = (double)other/base;
            sum_distance+=(other-base);
            assert(base != 0);
            sum_ratio += ratio;
            count++;

            max_distance = max(max_distance, distance);
            max_ratio = max(ratio, max_ratio);
        }

        string print()
        {
            return "sum_dist = \t" + std::to_string(sum_distance) +
                   "\t, max_dist = \t" + std::to_string(max_distance) +
                   "\t| avg_ratio = \t" + std::to_string(sum_ratio/count) +
                   "\t, max_ratio = \t" + std::to_string(max_ratio);
        }
    };

    class SynthesizerScoreSum
    {
    public:
        int sum_sizes = 0;

        void clear_score()
        {
            sum_sizes = 0;
        }

        string print()
        {
            return "sum = \t" + std::to_string(sum_sizes);
        }

        string simple_print()
        {
            return  std::to_string(sum_sizes);
        }

        void compose_with(DecisionTreeScore new_size)
        {
            sum_sizes+=new_size.size;
        }

        bool operator < (const SynthesizerScoreSum& other) const
        {
            return sum_sizes < other.sum_sizes;
        }
    };

    class SynthesizersScoreSummary: public NeuralNetwork
    {
    public:
        SynthesizerScoreSum base_synthesizer_score;
        SynthesizerScoreSum other_synthesizer_score;
        SynhtesizerScoreCoparitor comparitor;

        SynthesizersScoreSummary(){}

        SynthesizersScoreSummary(NeuralNetwork self): NeuralNetwork(self)
        {

        }

        void compose_with(DecisionTreeScore base_score, DecisionTreeScore other_score)
        {
            base_synthesizer_score.compose_with(base_score);
            other_synthesizer_score.compose_with(other_score);
            comparitor.compose_with(base_score, other_score);

        }

        int sum_errors()
        {
            return comparitor.sum_distance;
        }

        void clear_score()
        {
            base_synthesizer_score.clear_score();
            other_synthesizer_score.clear_score();
            comparitor.clear_score();
        }

        string print_other_synthesizer()
        {
            return "compare_with = {\t" + other_synthesizer_score.print() + "\t}";
        }

        string print()
        {
            return
                    "base = {\t" + base_synthesizer_score.print() +
                    "\t}; compare_with = {\t" + other_synthesizer_score.print()+
                    "\t}; comparitor = {\t" + comparitor.print() + "\t}";
        }

        string print_base()
        {
            return base_synthesizer_score.simple_print();
        }

        string print_other()
        {
            return other_synthesizer_score.simple_print();
        }

        bool operator < (const SynthesizersScoreSummary& other) const
        {
            return base_synthesizer_score < other.base_synthesizer_score;
        }
    };

    SynthesizersScoreSummary run_decision_tree_synthesis_comparison(
            SynthesizersScoreSummary meta_meta_net,
            BitvectorFunctionSolver<Data> bitvector_data,
            int n,
            vector<vector<Data> > data,
            bool run_reptile,
            bool print)
    {
        SynthesizersScoreSummary ret = meta_meta_net;
        ret.clear_score();
        for(int i = 0;i<data.size();i++)
        {
            SynthesizersScoreSummary meta_net = meta_meta_net;
            if(run_reptile)
            {
                evaluate_learner_parameters params;
                params.iter_cutoff = bitvector_data.leaf_iter;
                params.threshold = bitvector_data.min_treshold;


                reptile_train(meta_net, data[i], bitvector_data.root_iter, params);
            }

            for(int j = 0;j<data[i].size();j++)
            {
                /*SynthesizersScoreSummary leaf_net = meta_net;

                NeuralNetwork::parameters param = cutoff_param(8, 0.01);
                int local_iter = leaf_net.train(&data[i][j], param, &NeuralNetwork::softPriorityTrain);

                double local_error = leaf_net.get_error_of_data(&data[i][j]);

                if(print)
                {
                    cout << "f: " << data[i][j].printConcatinateOutput() << endl;
                    cout << "iters: " << local_iter <<endl;
                    cout << "error: " << local_error << endl;
                }*/


                SynthesizersScoreSummary new_leaf_net = meta_net;

                //tailor is the neural network that tailors towards inducing a special structure
                NeuralNetwork::parameters tailor_param = cutoff_param(7, 0.01);
                tailor_param.track_dimension_model = true;
                tailor_param.neural_net = &new_leaf_net;

                DecisionTreeSynthesisViaDP<Data> decision_tree_solver;
                DecisionTreeScore size;
                assert(0);
                /*
                 * NEED TO REFACTOR WAY OF CALLING THE DECISION TREE SYNTHESISER
                 * NEED TO PUT VALUE ON size
                 * =
                        decision_tree_solver.old_extract_decision_tree_and_compare_with_opt_and_entropy(tailor_param, n, data[i][j]);

                if(print)
                {
                    cout << "f: " << data[i][j].printConcatinateOutput() << endl;
                    cout << "opt = \t" << size.get_opt() << "\t; my = " << size.neural_guided_size << "\t; entropy = "
                         << size.entropy_guided_size << endl << endl;
                }

                if(size.opt_defined)
                {
                    assert(0);//update ret summary
                    / *if (size.neural_guided_size > size.get_opt()) {
                        ret.num_non_opt++;
                        ret.max_ratio = max(ret.max_ratio, (double) size.neural_guided_size / size.get_opt());
                        ret.max_delta = max(ret.max_delta, size.neural_guided_size - size.get_opt());
                        ret.sum_errors += size.neural_guided_size - size.get_opt();
                        ret.sum_sizes += size.neural_guided_size;

                        ret.non_opt.push_back(data[i][j]);
                    }
                    if (size.entropy_guided_size > size.get_opt()) {
                        ret.num_entropy_non_opt++;
                        ret.sum_entropy_errors += size.entropy_guided_size - size.get_opt();
                    }* /
                }
                else
                {
                    //need to handle case when opt is not defiend
                    assert(0);
                }*/
            }
        }
        return ret;
    }

    void SA_over_latent_decision_trees(
            SynthesizersScoreSummary &best,
            BitvectorFunctionSolver<Data> bitvector_data,
            int n,
            vector<vector<Data> > data,
            bool run_reptile_training)
    {
        double temperature = 1;
        int iter_count = 15;

        SynthesizersScoreSummary at = best = run_decision_tree_synthesis_comparison
                (best, bitvector_data, n, data, run_reptile_training, false);

        for(int iter = 0; iter<iter_count; iter++, temperature-=(1.0/iter_count))
        {
            //at.printWeights();

            SynthesizersScoreSummary next = NeuralNetworkAndScore(step(at, 0.35*temperature));

            //next.printWeights();

            SynthesizersScoreSummary next_score = run_decision_tree_synthesis_comparison
                    (next, bitvector_data, n, data, run_reptile_training, false);

            cout << "iter = " << iter << "; at:: " << at.print() <<"; next_score:: "<< next_score.print() <<endl;

            if(next_score < best)
            {
                at = next_score;
                best = next_score;
            }
            else if(next_score < at)
            {
                at = next_score;
            }
            else if(100*(double)(next_score.sum_errors()-at.sum_errors())/at.sum_errors() < (double)rand(0, (int)(25*temperature)))
            {
                cout << "take" <<endl;
                at = next_score;
            }
        }
    }

    void train(int n)
    {
        cout << "IN LatentDecisionTreeExtractor" <<endl <<endl;

        int leaf_iter = 30;

        BitvectorFunctionSolver<Data> bitvector_data(leaf_iter);

        //meta_net_and_score meta_meta_net = bitvector_data.train(n);
        SynthesizersScoreSummary meta_meta_net = NeuralNetworkAndScore(NeuralNetwork(n, 2*n, 1));

        assert(n<=4);
        bitvector_data.initMetaData(n, completeFirstOrderTestDataset<Data>(n));

        vector<vector<Data> > train_meta_data = bitvector_data.meta_data.train_meta_data;
        vector<vector<Data> > test_meta_data = bitvector_data.meta_data.test_meta_data;

        //vector<vector<Data> > harder_data = train_meta_data;

        /*for(int i = 0;i<1;i++)
        {
            vector<Data> step_harder_data = run_decision_tree_synthesis_comparison
                    (meta_meta_net, bitvector_data, n, harder_data, bitvector_data.treshold, false, false).non_opt;
            cout << step_harder_data.size() <<endl;
            harder_data = initSecondOrderDataset(step_harder_data, 2);
        }*/

        bool pre_train_meta_init = true;

        if(pre_train_meta_init) {

            bitvector_data.meta_data.train_meta_data.clear();
            int rand_step = train_meta_data.size()/3;
            for (int i = rand((int)(rand_step*0.5), (int)(1.5*rand_step));
                 i < train_meta_data.size();
                 i+=rand((int)(rand_step*0.5), (int)(1.5*rand_step)))
            {
                bitvector_data.meta_data.train_meta_data.push_back(train_meta_data[i]);
            }
            bitvector_data.meta_data.test_meta_data = train_meta_data;


            meta_meta_net = bitvector_data.train_to_meta_learn(n);
            bitvector_data.test_to_meta_learn(meta_net_and_score(meta_meta_net));

        }
        /*int prev_leaf_iter = bitvector_data.leaf_iter;
        for(bitvector_data.leaf_iter = 1; bitvector_data.leaf_iter < 16;bitvector_data.leaf_iter++) {
            SynthesizersScoreSummary local_meta_meta_net = meta_meta_net;
            SynthesizersScoreSummary at = run_decision_tree_synthesis_comparison
                    (local_meta_meta_net, bitvector_data, n, test_meta_data, pre_train_meta_init, false);

            cout << "leaf_iter = " << bitvector_data.leaf_iter << ", my_error = " << at.print() << endl;
            cout << "entropy_error = " << at.print_entropy() <<endl <<endl;
        }
        bitvector_data.leaf_iter = prev_leaf_iter;
        */
        SynthesizersScoreSummary at = run_decision_tree_synthesis_comparison
                (meta_meta_net, bitvector_data, n, test_meta_data, pre_train_meta_init, true);

        cout << "BEFORE TRAIN: at = " << at.print() << endl;
        cout << "Entropy = " << at.print_other_synthesizer() <<endl <<endl;

        SA_over_latent_decision_trees(meta_meta_net, bitvector_data, n, train_meta_data, pre_train_meta_init);

        cout << "In main train..)" <<endl;
        cout <<"meta_meta_net:: " << meta_meta_net.print() <<endl;


        if(pre_train_meta_init)
        {
            cout << "meta_meta_net after train:" << endl;
            bitvector_data.test_to_meta_learn(meta_net_and_score(meta_meta_net));
        }
        SynthesizersScoreSummary final_rez = run_decision_tree_synthesis_comparison
                (meta_meta_net, bitvector_data, n, test_meta_data, pre_train_meta_init, true);

        cout << "AFTER TRAIN: at = " << final_rez.print() << endl;
        cout << "entropy = " << final_rez.print_other_synthesizer() <<endl <<endl;

        cout << "END" <<endl;


    }


    /*SynthesizersScoreSummary run_new_decision_tree_synthesis_comparison(
            vector<NeuralNetwork*> progressive_nets,
            BitvectorFunctionSolver bitvector_data,
            int n,
            vector<vector<Data> > data,
            bool print)
    {
        SynthesizersScoreSummary ret;
        ret.clear_score();

        for(int i = 0;i<data.size();i++)
        {
            for(int j = 0;j<data[i].size();j++)
            {
                NeuralNetwork::parameters tailor_param = meta_cutoff(bitvector_data.root_iter, bitvector_data.leaf_iter);
                tailor_param.progressive_nets = progressive_nets;

                DecisionTreeSynthesisViaDP decision_tree_solver;
                DecisionTreeScore size =
                        decision_tree_solver.new_extract_decision_tree_and_compare_with_opt_and_entropy(tailor_param, n, data[i][j]);

                if(print)
                {
                    cout << "f: " << data[i][j].printConcatinateOutput() << endl;
                    cout << "opt = \t" << size.opt << "\t; my = " << size.neural_guided_size << "\t; entropy = "
                         << size.entropy_guided_size << endl << endl;
                }

                if(size.neural_guided_size > size.opt)
                {
                    ret.num_non_opt++;
                    ret.max_ratio = max(ret.max_ratio, (double)size.neural_guided_size/size.opt);
                    ret.max_delta = max(ret.max_delta, size.neural_guided_size - size.opt);
                    ret.sum_errors += size.neural_guided_size - size.opt;
                    ret.sum_sizes += size.neural_guided_size;

                    ret.non_opt.push_back(data[i][j]);
                }
                if(size.entropy_guided_size > size.opt)
                {
                    ret.num_entropy_non_opt++;
                    ret.sum_entropy_errors += size.entropy_guided_size - size.opt;
                }
            }
        }
        return ret;
    }*/

    //find NN init s.t.
    //After training a training set on sort ERROR(NN, f) = sort |DT_opt(f)|
    //it can solve bigger and bigger parts of a validation set ERROR(NN, f) = sort |DT_opt(f)|

    //NEED A WAY TO GENERATE TRAINING SETS AND VALIDATION SETS


    //find NN init s.t.
    //After training on a training set of (backprop on ERROR(NN, f) = sort |DT_opt(f)|, evolution on ERROR(NN, f) = sort |DT_opt(f)|) as before
    //After training a training set on sort ERROR(NN, f) = sort |DT_opt(f)|
    //it can solve bigger and bigger parts of a validation set ERROR(NN, f) = sort |DT_opt(f)|

    //NEED A WAY TO GENERATE TRAINING SETS AND VALIDATION SETS


    //The goal of training is #not to forget
    //find NN initialization, such that training on a tasks forgets minimally == few shot learning


    //use k instead of error...
    //so goal is to minimize sum k, while maximizing blank space in between endpoints


    //What do we need for curriculum learning?
    //We need a tested training scheme
    //Calculate a distribution from which to sample, which ids to learn and unlearn
    //Need a representation of a distribution.
    //unlearn, proportional to # swaps necessary to put in place? if 0 swaps, then learn;

    //parameters needed for learning:



    template<typename DataType>
    void train_library(
            int min_trainable_n,
            int max_n, vector<int> leaf_iters, vector<int> hidden_layer_width, vector<int> hidden_layer_depth,
            int num_ensambles, vector<FirstOrderDataset<DataType> > first_order_datasets,
            typename FirstOrderLearning<DataType>::learning_parameters all_params)
    {
        assert(leaf_iters.size() > max_n-1);
        assert(hidden_layer_width.size() > max_n-1);
        NeuralNetworkAndScore ensamble_progressive_nets[10][10];
        vector<vector<NeuralNetworkAndScore*> > ensamble_progressive_net_pointers;
        ensamble_progressive_net_pointers.push_back(vector<NeuralNetworkAndScore*>());// for n = 0;

        for(int local_n = 2; local_n < min_trainable_n; local_n++)
        {
            vector<NeuralNetworkAndScore*> local_ensamble;
            for(int ensamble_id = 0; ensamble_id < num_ensambles; ensamble_id++) {

                vector<int> hidden_layer_widths = hidden_layer_block(
                        hidden_layer_width[local_n - 1], hidden_layer_depth[local_n-1]);

                NeuralNetworkAndScore meta_meta_net =
                        NeuralNetworkAndScore(NeuralNetwork(local_n - 1, hidden_layer_widths, 1));

                ensamble_progressive_nets[local_n - 1][ensamble_id] = meta_meta_net;
                local_ensamble.push_back(&ensamble_progressive_nets[local_n - 1][ensamble_id]);
            }
            ensamble_progressive_net_pointers.push_back(local_ensamble);
        }

        for(int local_n = min_trainable_n; local_n <= max_n; local_n++)
        {


//            cout << "HERE A-2" << endl;

            vector<NeuralNetworkAndScore*> local_ensamble =
                    get_local_ensamble(
                            local_n-1, hidden_layer_width, hidden_layer_depth,
                            num_ensambles, first_order_datasets, ensamble_progressive_nets, all_params);

            // (local_n, leaf_iters, hidden_layer_width, hidden_layer_depth,
            //       num_ensambles, first_order_datasets, do_learn, ensamble_progressive_net_pointers);

            ensamble_progressive_net_pointers.push_back(local_ensamble);

            NeuralNetwork::parameters cutoff_parameter = NeuralNetwork::parameters(leaf_iters);
            cutoff_parameter.priority_train_f = all_params.choosePriorityTrain;
            cutoff_parameter.progressive_ensamble_nets = ensamble_progressive_net_pointers;

            bool do_learn = true;

////            need to fix; pre-training model parameters differ from testing model parameters. Check how reptile_step_vary_epsilon is done
//            do_learn = true;
//            for(int i = 0;i<cutoff_parameter.progressive_ensamble_nets[local_n-1].size();i++)
//            {
//                if(cutoff_parameter.progressive_ensamble_nets[local_n-1][i]->sum_ordering_error != 0)
//                {
//                    do_learn = false;
//                }
//            }
//
//            cout << "DO_LEARN = " << do_learn << endl;
//
            if(do_learn)
            {
                cutoff_parameter.progressive_ensamble_neural_error_buckets =
                        get_progressive_ensamble_neural_error_buckets
                                (local_n - 1, first_order_datasets, cutoff_parameter);
            }

            BitvectorFunctionSolver<DataType> complete_bitvector_data(leaf_iters[local_n]);

            complete_bitvector_data.initFirstOrderData(local_n, first_order_datasets[local_n]);

            DecisionTreeSynthesisViaDP<DataType> decision_tree_solver;

            SynthesizersScoreSummary opt_to_entropy;
            SynthesizersScoreSummary opt_to_combined;
            SynthesizersScoreSummary opt_to_neural;
            SynthesizersScoreSummary opt_to_random;
            SynthesizersScoreSummary neural_to_entropy;
            SynthesizersScoreSummary random_to_entropy;


            for(int i = 0;i < complete_bitvector_data.first_order_data.test_data.size();i++) {

                DecisionTreeScore size_opt, size_combined_guided, size_neural_guided, size_entropy_guided, size_random_guided;

                if(local_n <= 4)
                {
                    size_opt = decision_tree_solver.synthesize_decision_tree_and_get_size
                            (cutoff_parameter, local_n, complete_bitvector_data.first_order_data.test_data[i], optimal);
                }

                size_combined_guided =
                size_neural_guided = decision_tree_solver.synthesize_decision_tree_and_get_size
                        (cutoff_parameter, local_n, complete_bitvector_data.first_order_data.test_data[i], combined_guided);

                size_neural_guided = decision_tree_solver.synthesize_decision_tree_and_get_size
                        (cutoff_parameter, local_n, complete_bitvector_data.first_order_data.test_data[i], neural_guided);

                size_entropy_guided = decision_tree_solver.synthesize_decision_tree_and_get_size
                        (cutoff_parameter, local_n, complete_bitvector_data.first_order_data.test_data[i], entropy_guided);

                //size_random_guided = decision_tree_solver.synthesize_decision_tree_and_get_size
                //        (cutoff_parameter, local_n, complete_bitvector_data.first_order_data.test_data[i], random_guided);

                //cout parameters:
                //cout << complete_bitvector_data.first_order_data.test_data[i].printConcatinateOutput() << endl;
                //cout << decision_tree_size.print() << endl << endl;

                if(local_n <= 4)
                {
                    opt_to_entropy.compose_with(size_opt, size_entropy_guided);
                    opt_to_neural.compose_with(size_opt, size_neural_guided);
                    //opt_to_random.compose_with(size_opt, size_random_guided);
                }

                neural_to_entropy.compose_with(size_neural_guided, size_entropy_guided);
                //random_to_entropy.compose_with(size_random_guided, size_entropy_guided);

            }

            cout << endl;

            fout << num_ensambles << "\t" << opt_to_neural.print_base() << "\t" << neural_to_entropy.print_base() <<"\t" << neural_to_entropy.print_other() <<endl;

            cout << "ensamble_size = " << num_ensambles << endl;
            cout << "hidden_layer_width:" <<endl;
            for(int j = 2; j < local_n; j++)
            {
                cout << "n_in = " << j << "\thidden_width "<< hidden_layer_width[j] <<endl;
            }
            cout << "leaf_iters:" <<endl;
            for(int j = 2; j < local_n; j++)
            {
                cout << "n_in = " << j << "\tleaf_iter "<< leaf_iters[j] <<endl;
            }
            if(local_n <= 4 && true)
            {
                cout << "opt_to_neural:: " << opt_to_neural.print() << endl;
                //cout << "opt_to_entropy:: " << opt_to_entropy.print() << endl;
                //cout << "opt_to_random:: " << opt_to_random.print() << endl;
            }
            cout << "neural_to_entropy:: " << neural_to_entropy.print() << endl;
            //cout << "random_to_entropy:: " << random_to_entropy.print() << endl <<endl;
        }
    }
};


#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_LATENTDECISIONTREEEXTRACTOR_H
