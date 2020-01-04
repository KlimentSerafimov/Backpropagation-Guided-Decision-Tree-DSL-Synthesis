//
// Created by Kliment Serafimov on 2020-01-02.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_SECONDORDERLEARNING_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_SECONDORDERLEARNING_H

#include <vector>
#include "FirstOrderLearning.h"

using namespace std;

template<typename DataType>
class SecondOrderLearning: public FirstOrderLearning<DataType>
{
public:

    SecondOrderLearning(): FirstOrderLearning<DataType>()
    {

    }

    void learn_to_meta_learn(
            meta_net_and_score &learner,
            vector<vector<Data> > train_meta_data,
            int root_iter, typename FirstOrderLearning<DataType>::evaluate_learner_parameters params)
    {
        learning_to_reptile(learner, train_meta_data, root_iter, params);
    }

    void learning_to_reptile(
            meta_net_and_score &global_best_solution,
            vector<vector<Data> > f_data,
            int root_iter_init,
            typename FirstOrderLearning<DataType>::evaluate_learner_parameters params)
    {
        int leaf_iter_init = params.iter_cutoff;
        double treshold = params.threshold;
        double max_treshold = treshold;
        double min_treshold = max_treshold/3;

        int root_iter = root_iter_init;
        int k_iter = params.iter_cutoff;

        int global_stagnation = 0;

        meta_net_and_score at_walking_solution;

        meta_net_and_score best_solution;

        meta_net_and_score SA_iter_best_solution = best_solution = at_walking_solution =
                evaluate_meta_learner(global_best_solution, f_data, true, root_iter, params);

        if(best_solution < global_best_solution)
        {
            global_best_solution = best_solution;
        }

        bool enter = false;

        int give_up_after = 1;

        for(int k_iter_loops = 0; global_stagnation < give_up_after; )
        {

            if(enter)
            {
                treshold*=0.8;
                treshold = max(treshold, min_treshold);
                if(treshold == min_treshold)
                {
                    k_iter--;
                }
                at_walking_solution = evaluate_meta_learner(SA_iter_best_solution, f_data, true, root_iter, params);
            }
            cout << "NEW k_iter = " << k_iter <<"; NEW treshold = " << treshold << endl;

            enter = true;

            int count_stagnation = 0;

            int SA_stagnation = 0;

            assert(k_iter>0);

            for (int iter = 1; at_walking_solution.num_train_fail > 0 && global_stagnation < give_up_after; iter++) {

                meta_reptile_step(at_walking_solution, f_data, root_iter, params);

                //at_walking_solution.printWeights();

                int repeat_const = 2;

                int repeat_count = 1+0*(repeat_const*f_data.size());

                if (iter % repeat_count == 0) {

                    k_iter_loops+=repeat_count;
                    NeuralNetworkAndScore new_score = at_walking_solution =
                                                      evaluate_meta_learner(at_walking_solution, f_data, true, root_iter, params);

                    //cout << new_score << endl;

                    //at_walking_solution.printWeights();

                    if (new_score < SA_iter_best_solution) {
                        SA_iter_best_solution = new_score;
                        count_stagnation = 0;

                        FirstOrderLearning<DataType>::update_solutions
                                (SA_iter_best_solution, best_solution, global_best_solution,
                                 global_stagnation, SA_stagnation);
                    }
                    else
                    {
                        count_stagnation++;
                        SA_stagnation++;
                        if(count_stagnation >= 1+log2(f_data.size()) || SA_stagnation >= 1+2*log2(f_data.size()))
                        {

                            double radius = 16*SA_iter_best_solution.max_error/(f_data[0][0].size()*f_data[0].size()*f_data.size());
                            NeuralNetwork next_step = FirstOrderLearning<DataType>::step(at_walking_solution, radius);
                            cout << "SA step" << endl;
                            new_score = SA_iter_best_solution = at_walking_solution =
                                    evaluate_meta_learner(next_step, f_data, true, root_iter, params);

                            count_stagnation=0;
                            global_stagnation++;
                            k_iter = leaf_iter_init;


                            FirstOrderLearning<DataType>::update_solutions
                                    (SA_iter_best_solution, best_solution, global_best_solution,
                                     global_stagnation, SA_stagnation);
                        }
                    }

                    cout << "local stangation = " << count_stagnation <<endl;
                    cout << "global stangation = " << global_stagnation <<endl;
                    cout << "k_iter = " << k_iter << endl;
                    cout << "k_iter_loops = " << k_iter_loops << endl;
                    cout << iter << " at            = " << new_score.print() << endl;
                    cout << iter << " SA local best = " << SA_iter_best_solution.print() << endl;
                    cout << iter << " reptile  best = " << best_solution.print() << endl;

                    tout << new_score.clean_print() << endl;

                }
            }
        }
    }

    void meta_reptile_step(
            NeuralNetwork &try_learner,
            vector<vector<Data> > data,
            int root_cutoff,
            typename FirstOrderLearning<DataType>::evaluate_learner_parameters params)
    {
        int i;

        i = rand(0, data.size()-1);

        {
            //cout << "train on task: " << i <<endl;
            //try_learner.printWeights();
            NeuralNetwork local_try_learner = NeuralNetwork(try_learner.copy());
            //NeuralNetworkAndScore before_train = evaluate_learner(local_try_learner, data[i], print, leaf_cutoff);

            FirstOrderLearning<DataType>::reptile_train(local_try_learner, data[i], root_cutoff, params);

            local_try_learner.minus(try_learner);
            local_try_learner.mul(-1.0/data.size());
            try_learner.minus(local_try_learner);
        }

        //cout << "average = " << (double)sum_train_iter/Dataset_of_functions.size() <<endl;
    }

    meta_net_and_score evaluate_meta_learner(
            NeuralNetwork try_learner,
            vector<vector<Data> > data,
            bool print,
            int root_iter,
            typename FirstOrderLearning<DataType>::evaluate_learner_parameters params)
    {
        meta_net_and_score score = NeuralNetworkAndScore(try_learner);

        score.clear_vals();

        if(print)
        {
            cout << "IN evaluate_meta_learner" << endl;
        }
        for(int i = 0;i<data.size();i++)
        {
            //cout << i <<endl;
            NeuralNetwork local_try_learner = NeuralNetwork(try_learner.copy());


            FirstOrderLearning<DataType>::reptile_train(local_try_learner, data[i], root_iter, params);

            NeuralNetworkAndScore reptile_score =
                    FirstOrderLearning<DataType>::evaluate_learner_softPriorityTrain(local_try_learner, data[i], print, params);

            score.is_init_score = false;
            score.max_error = max(score.max_error, reptile_score.max_error);
            score.sum_error += reptile_score.sum_error;
            score.num_train_fail += reptile_score.num_train_fail;
            score.max_leaf_iter = max(score.max_leaf_iter, reptile_score.max_leaf_iter);

            if(print)
            {
                cout << "\t" << i << "\t" << reptile_score.max_error << endl;
                //assert(0);//do the comment
                //tmp.pring_delta_w(try_learner);
                //local_try_learner.printWeights();
            }
        }
        if(print) {
            cout << "END evaluate_meta_learner" << endl;
        }
        //cout << "average = " << (double)sum_train_iter/Dataset_of_functions.size() <<endl;
        return score;
    }
};


#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_SECONDORDERLEARNING_H
