//
// Created by Kliment Serafimov on 2019-02-24.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_FIRSTORDERLEARNING_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_FIRSTORDERLEARNING_H

#include "NeuralNetwork.h"
#include "Data.h"
#include "Header.h"
#include "NeuralNetworkAndScore.h"
#include "Policy.h"

static ofstream tout("trace.out");

template<typename datatype>
class FirstOrderLearning
{
public:

    FirstOrderLearning()
    {

    }

    NeuralNetwork step(NeuralNetwork prev_step, double rate)
    {
        //prev_step.printWeights();
        NeuralNetwork ret = NeuralNetwork(prev_step);
        ret.perturb(rate);
        //ret.printWeights();
        return ret;
    }

    NeuralNetwork deep_step(NeuralNetwork prev_step, double rate) {
        int backtrack_potential = 3;

        int search_width = 3;

        // if search_width steps are is backtrack take best step; if backtrack do it only if: (1) if [ [sum_train_iter of # of candidates that are better than proposed step] OR absolute difference] < [backtrack potential]
        // and (2) second attempt from current learner;

        // track somehow #of itterations steps forward (have to take into account how many other threads reached that perticular hight) vs #itteration steps cost vs #iteration steps backwards (same as before)

        // track progress of roots via their kids.


        //need one big priority queue with stocastic swarm search with decaying tree like search that atributes credit to branches that have shown/are showing progress;



        //BUT ALSO JUST SEE WHAT ARE THE DELTAS!! DONE


        //ALSO DO DP ON THE STRUCTURE OF THE NEURAL NETWORK. NO NEED FOR ALL CONNECTED

        return NeuralNetwork();
    }

    /*void progressive_train(int n)
    {
        NeuralNetworkAndScore learner = NeuralNetwork(n, 2*n, 1);

        int k_iter = 30;
        //performs meta training, using the max error after k iter of a task as signal.


        vector<Datatype> local_data;

        for(int i = 0;i<data_of_functions.size();)
        {
            int data_step_size = 5;
            for(int k = 0; k < data_step_size && i<data_of_functions.size();i++, k++)
            {
                local_data.push_back(data_of_functions[i]);
            }
            reptile_SA_train(learner, local_data, k_iter);
            evaluate_learner(learner, test_data_of_functions, true, 300);
        }

    }*/
/*
    void custom_distribution_reptile_step(
            NeuralNetwork &try_learner,
            errorAsClassificatorData data,
            int iter_cutoff,
            double treshold)
    {
        //for(int i = 0;i<data.size();i++)
        int i;

        i = rand(0, data.size()-1);

        {
            NeuralNetwork local_try_learner = NeuralNetwork(try_learner.copy());
            NeuralNetwork::parameters learning_param = cutoff_param(iter_cutoff, 0.01);
            int num_iter = local_try_learner.train(&data[i], learning_param., &NeuralNetwork::softPriorityTrain);

            double local_error = local_try_learner.get_error_of_data(&data[i]);
            local_try_learner.minus(try_learner);
            local_try_learner.mul(-1.0/data.size());
            try_learner.minus(local_try_learner);
        }

    }
    
    void custom_distribution_meta_learn(
            NeuralNetworkAndScore &global_best_solution,
            errorAsClassificatorData custom_data,
            int k_iter_init,
            double treshold)
    {

        int k_iter = k_iter_init;

        int global_stagnation = 0;

        NeuralNetworkAndScore at_walking_solution;

        NeuralNetworkAndScore best_solution;

        NeuralNetworkAndScore SA_iter_best_solution = best_solution = at_walking_solution =
                evaluate_learner(global_best_solution, data, false, k_iter, treshold);

        for(int k_iter_loops = 0; global_stagnation < 4; k_iter--)
        {
            cout << "NEW k_iter = " << k_iter << endl;
            at_walking_solution = evaluate_learner(SA_iter_best_solution, data, false, k_iter, treshold);

            int count_stagnation = 0;

            assert(k_iter>0);
            for (int iter = 1; at_walking_solution.num_train_fail > 0 && global_stagnation < 4; iter++) {

                custom_distribution_reptile_step(at_walking_solution, data, k_iter, treshold);



                //at_walking_solution.printWeights();

                int repeat_const = 3;

                int repeat_count = (repeat_const*data.size());

                if (iter % repeat_count == 0) {

                    k_iter_loops+=repeat_count;
                    NeuralNetworkAndScore new_score = at_walking_solution = evaluate_learner(at_walking_solution, data, true, k_iter, treshold);

                    //cout << new_score << endl;

                    //at_walking_solution.printWeights();


                    if (new_score < SA_iter_best_solution) {
                        SA_iter_best_solution = new_score;
                        count_stagnation = 0;
                        if(SA_iter_best_solution < best_solution)
                        {
                            global_stagnation = 0;
                            best_solution = SA_iter_best_solution;
                            if(best_solution < global_best_solution)
                            {
                                global_best_solution = best_solution;
                            }
                        }
                    }
                    else
                    {
                        count_stagnation++;
                        if(count_stagnation >= log2(data.size()))
                        {
                            NeuralNetwork next_step = step(at_walking_solution, 16*SA_iter_best_solution.max_error/(data[0].size()*data.size()));
                            cout << "SA step" << endl;
                            new_score = SA_iter_best_solution = at_walking_solution = evaluate_learner(next_step, data, true, k_iter, treshold);

                            count_stagnation=0;
                            global_stagnation++;
                            k_iter = k_iter_init;

                            if(SA_iter_best_solution < best_solution)
                            {
                                global_stagnation = 0;
                                best_solution = SA_iter_best_solution;
                                if(best_solution < global_best_solution)
                                {
                                    global_best_solution = best_solution;
                                }
                            }
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
*/

    /*void reptile_SA_train(
            NeuralNetworkAndScore &global_best_solution,
            vector<Datatype> data,
            int k_iter_init,
            double treshold)
    {
        k_iter_init = 60;

        int k_iter = k_iter_init;

        int global_stagnation = 0;

        NeuralNetworkAndScore at_walking_solution;

        NeuralNetworkAndScore best_solution;

        NeuralNetworkAndScore SA_iter_best_solution = best_solution = at_walking_solution =
                evaluate_learner(global_best_solution, data, false, k_iter, treshold);

        for(int k_iter_loops = 0; global_stagnation < 4; k_iter--)
        {
            cout << "NEW k_iter = " << k_iter << endl;
            at_walking_solution = evaluate_learner(SA_iter_best_solution, data, false, k_iter, treshold);

            int count_stagnation = 0;

            assert(k_iter>0);
            for (int iter = 1; at_walking_solution.num_train_fail > 0 && global_stagnation < 4; iter++) {

                reptile_step(at_walking_solution, data, k_iter, treshold);

                //at_walking_solution.printWeights();

                int repeat_const = 3;

                int repeat_count = (repeat_const*data.size());

                if (iter % repeat_count == 0) {

                    k_iter_loops+=repeat_count;
                    NeuralNetworkAndScore new_score = at_walking_solution = evaluate_learner(at_walking_solution, data, true, k_iter, treshold);

                    //cout << new_score << endl;

                    //at_walking_solution.printWeights();


                    if (new_score < SA_iter_best_solution) {
                        SA_iter_best_solution = new_score;
                        count_stagnation = 0;
                        if(SA_iter_best_solution < best_solution)
                        {
                            global_stagnation = 0;
                            best_solution = SA_iter_best_solution;
                            if(best_solution < global_best_solution)
                            {
                                global_best_solution = best_solution;
                            }
                        }
                    }
                    else
                    {
                        count_stagnation++;
                        if(count_stagnation >= log2(data.size()))
                        {
                            NeuralNetwork next_step = step(at_walking_solution, 16*SA_iter_best_solution.max_error/(data[0].size()*data.size()));
                            cout << "SA step" << endl;
                            new_score = SA_iter_best_solution = at_walking_solution = evaluate_learner(next_step, data, true, k_iter, treshold);

                            count_stagnation=0;
                            global_stagnation++;
                            k_iter = k_iter_init;

                            if(SA_iter_best_solution < best_solution)
                            {
                                global_stagnation = 0;
                                best_solution = SA_iter_best_solution;
                                if(best_solution < global_best_solution)
                                {
                                    global_best_solution = best_solution;
                                }
                            }
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
    }*/


    bool update_solutions(NeuralNetworkAndScore &SA_iter_best_solution, NeuralNetworkAndScore &best_solution,
                          NeuralNetworkAndScore &global_best_solution,
                          int &global_stagnation, int &SA_stagnation)
    {
        if(SA_iter_best_solution < best_solution)
        {
            SA_stagnation = 0;

            update_net_and_score(best_solution, SA_iter_best_solution);

            cout << "NEW BEST SOLUTION" <<endl;

            if(best_solution < global_best_solution)
            {
                global_stagnation = 0;

                update_net_and_score(global_best_solution, best_solution);

            }
            return true;
        }
        return false;
    }

    class learning_parameters
    {
    public:
        int give_up_after;

        double init_epsilon;

        double end_epsilon;

        int repeat_count;
        int max_num_evals;
        int max_num_iter;

        double trials;

        double step_radius_size;

        double init_temperature;
        double end_temperature;

        NeuralNetwork::PriorityTrainReport (NeuralNetwork::*choosePriorityTrain)(Data*, NeuralNetwork::parameters);

        int leaf_iter_init;

        double treshold;

        int max_boundary_size;

        learning_parameters()
        {

        }

        void at_initiation_of_parameters_may_21st_10_52_pm()
        {
            choosePriorityTrain = &NeuralNetwork::softPriorityTrain;
            give_up_after = 50;

            init_epsilon = 0.66;
            end_epsilon = 0.09;

            repeat_count = 1;
            max_num_iter = 30;

            max_num_evals = max_num_iter;


            trials = 5;

            step_radius_size = 0.1;

            init_temperature = 1;
            end_temperature = 0;
        }
    };

    class evaluate_learner_parameters
    {
    public:
        int iter_cutoff;
        int exstended_iter_cutoff;
        double threshold;

        evaluate_learner_parameters(){}

        evaluate_learner_parameters(learning_parameters copy_from)
        {
            iter_cutoff = copy_from.leaf_iter_init;
            threshold = copy_from.treshold;
        }
    };


    void reptile_SA_train(
            NeuralNetworkAndScore &global_best_solution,
            vector<datatype> f_data,
            learning_parameters init_param,
            bool print)
    {
        //OLD o_rder_neural_errors with a trainer that learns to reduce k for the normal tasks
        learning_parameters learning_param = init_param;

        double max_treshold = learning_param.treshold;
        double min_treshold = (2*max_treshold)/3;

        int k_iter = learning_param.leaf_iter_init;

        int global_stagnation = 0;

        Policy local_policy(f_data, false);
        local_policy.policy_type = Policy::total_drift;

        non_dominating_score_boundary boundary;

        NeuralNetworkAndScore at_walking_solution;

        NeuralNetworkAndScore best_solution;

        NeuralNetworkAndScore SA_iter_best_solution = best_solution = at_walking_solution =
                evaluate_learner(global_best_solution, f_data, false,
                        k_iter,learning_param.treshold, learning_param.choosePriorityTrain);

        local_policy.update(&at_walking_solution);

        //print_ordering(local_policy, f_data, at_walking_solution.tarjans);

        boundary.update(at_walking_solution);


        if(best_solution < global_best_solution)
        {
            global_best_solution = best_solution;
        }

        bool enter = false;

        double epsilon = learning_param.init_epsilon;

        int repeat_count =learning_param.repeat_count;
        int max_num_evals =learning_param.max_num_evals;
        int max_num_iter = max_num_evals*repeat_count;


        for(int k_iter_loops = 0; global_stagnation <learning_param.give_up_after
                                  && epsilon >=learning_param.end_epsilon; )
        {


            if(enter)
            {
                local_policy.policy_type = Policy::controled_drift;

               learning_param.treshold*=0.8;
                if(learning_param.treshold < min_treshold)
                {
                    if(at_walking_solution.sum_ordering_error == 0)
                    {
                        break;
                    }
                }
                else
                {
                   learning_param.treshold = max(learning_param.treshold, min_treshold);
                }

                if(learning_param.treshold == min_treshold)
                {
                    k_iter *= 4;
                    k_iter /= 5;

                    k_iter = max(k_iter, 2*f_data[0].size());
                }
                at_walking_solution = evaluate_learner(SA_iter_best_solution, f_data, false, k_iter,learning_param.treshold,learning_param.choosePriorityTrain);
                local_policy.update(&at_walking_solution);
                //print_ordering(local_policy, f_data, at_walking_solution.tarjans);

            }
            if(print)
            {
                cout << "NEW k_iter = " << k_iter <<"; NEW treshold = " <<learning_param.treshold << endl;
            }

            enter = true;

            int count_stagnation = 0;

            int SA_stagnation = 0;


            assert(k_iter>0);

            for (

                    int iter = 1;

                    (at_walking_solution.sum_ordering_error > 0
                     || at_walking_solution.num_train_fail > 0)
                    && global_stagnation <learning_param.give_up_after
                    && epsilon >=learning_param.end_epsilon
                    ;

                    iter++

                    ) {


                epsilon -= (-learning_param.end_epsilon+learning_param.init_epsilon)/max_num_iter;

                //epsilon = 0.05+(double)(at_walking_solution.ordering_error)/300;

                //at_walking_solution = boundary.query();

                reptile_step_vary_epsilon(at_walking_solution, f_data, k_iter, local_policy, epsilon,learning_param.choosePriorityTrain);

                //at_walking_solution.printWeights();


                if (iter % repeat_count == 0) {

                    k_iter_loops+=repeat_count;

                    cout << k_iter_loops << " / "<< max_num_iter << endl;

                    at_walking_solution = evaluate_learner(at_walking_solution, f_data, print, k_iter,learning_param.treshold,learning_param.choosePriorityTrain);

                    local_policy.update(&at_walking_solution);

                    //print_ordering(local_policy, f_data, at_walking_solution.tarjans);

                    if(true || at_walking_solution.max_error < 0.499)
                    {

                        local_policy.policy_type = Policy::controled_drift;
                    }
                    else{

                        local_policy.policy_type = Policy::total_drift;
                    }

                    //cout << new_score << endl;

                    //at_walking_solution.printWeights();

                    boundary.update(at_walking_solution);

                    if(at_walking_solution < SA_iter_best_solution) {

                        update_net_and_score(SA_iter_best_solution, at_walking_solution);

                        count_stagnation = 0;

                        if(update_solutions
                                (SA_iter_best_solution, best_solution, global_best_solution,
                                 global_stagnation, SA_stagnation))
                        {
                            print_stats
                                    (epsilon, count_stagnation, global_stagnation, iter, k_iter, k_iter_loops, at_walking_solution, SA_iter_best_solution, best_solution, boundary, local_policy, f_data);
                        }
                    }
                    else
                    {


                        count_stagnation++;
                        SA_stagnation++;



                        if(count_stagnation >=learning_param.trials)
                        {
                            global_stagnation++;


                            NeuralNetwork next_step = step(SA_iter_best_solution,learning_param.step_radius_size);

                            cout << "SA step" << endl;


                            NeuralNetworkAndScore try_step_solution = evaluate_learner(next_step, f_data, print, k_iter,learning_param.treshold,learning_param.choosePriorityTrain);

                            k_iter =learning_param.leaf_iter_init;

                            if(false) {

                                if (boundary.update(try_step_solution)) {

                                    cout << "SUCCESS" << endl;

                                    local_policy.update(&try_step_solution);
                                    print_ordering(local_policy, f_data, try_step_solution.tarjans);

                                    update_net_and_score(at_walking_solution, try_step_solution);

                                    update_net_and_score(SA_iter_best_solution, at_walking_solution);
                                    count_stagnation = 0;

                                    if(update_solutions
                                            (SA_iter_best_solution, best_solution, global_best_solution,
                                             global_stagnation, SA_stagnation))
                                    {
                                        print_stats(epsilon, count_stagnation, global_stagnation, iter, k_iter, k_iter_loops, at_walking_solution, SA_iter_best_solution, best_solution, boundary, local_policy, f_data);
                                    }
                                }
                                else
                                {
                                    cout << "TO TRY AGAIN" << endl;
                                }
                            }

                            if(true) {

                                count_stagnation = 0;

                                boundary.update(try_step_solution);
                                local_policy.update(&try_step_solution);

                                //print_ordering(local_policy, f_data, try_step_solution.tarjans);

                                update_net_and_score(at_walking_solution, try_step_solution);

                                update_net_and_score(SA_iter_best_solution, at_walking_solution);

                                if(update_solutions
                                        (SA_iter_best_solution, best_solution, global_best_solution,
                                         global_stagnation, SA_stagnation))
                                {
                                    print_stats(epsilon, count_stagnation, global_stagnation, iter, k_iter, k_iter_loops, at_walking_solution, SA_iter_best_solution, best_solution, boundary, local_policy, f_data);
                                }
                            }
                        }

                    }

                    if(false) {

                        print_stats(epsilon, count_stagnation, global_stagnation, iter, k_iter, k_iter_loops, at_walking_solution, SA_iter_best_solution, best_solution, boundary, local_policy, f_data);

                    }
                }
                else{

                    boundary.update(at_walking_solution);
                }
            }
        }
    }

    void meta_learn(
            NeuralNetworkAndScore &global_best_solution,
            vector<datatype> f_data,
            learning_parameters learning_param,
            bool print)
    {
        reptile_SA_train(global_best_solution, f_data, learning_param, print);
    }

    bool explore_step(
            NeuralNetworkAndScore at_walking_solution, NeuralNetworkAndScore try_solution, double temperature, bool print)
    {
        assert(try_solution.sum_ordering_error >= at_walking_solution.sum_ordering_error);

        if(print)
            cout << "attempts perturb(from " << at_walking_solution.sum_ordering_error  << " to " << try_solution.sum_ordering_error << ") ";
        if(100*(0.75-(((1+try_solution.sum_ordering_error)/(1+at_walking_solution.sum_ordering_error))-1)) > rand(0, 100))
        {
            if(print)
                cout << "PETURBS" << endl;
            return true;
        }
        else
        {
            if(print)
                cout << endl;
        }
        return false;
    }

    NeuralNetworkAndScore order_neural_errors(
            NeuralNetworkAndScore global_best_solution,
            vector<datatype> f_data,
            learning_parameters init_param,
            bool print)
    {
        learning_parameters learning_param = init_param;

        int k_iter =learning_param.leaf_iter_init;

        evaluate_learner_parameters eval_param = evaluate_learner_parameters(learning_param);

        Policy local_policy = Policy(f_data);

        non_dominating_score_boundary boundary;

        boundary.set_max_boundary_size(learning_param.max_boundary_size);

        NeuralNetworkAndScore at_walking_solution =
                evaluate_learner(global_best_solution, f_data, false, eval_param,learning_param.choosePriorityTrain);

        local_policy.update(&at_walking_solution);

        global_best_solution = at_walking_solution;

        //print_ordering(local_policy, f_data, at_walking_solution.tarjans);

        boundary.update(at_walking_solution);


        double epsilon =learning_param.init_epsilon;
        double temperature =learning_param.init_temperature;

        int max_num_iter = learning_param.max_num_iter;

        if(print)
        {
            print_stats(epsilon, temperature, 0, k_iter, at_walking_solution, global_best_solution,
                        boundary, local_policy, f_data);
        }

        for (int iter = 1;
                iter <= max_num_iter &&
                epsilon >=learning_param.end_epsilon;
                iter++) {

            if(print)
            {
                cout << iter << " / " << max_num_iter << endl;
            }

            epsilon = max(learning_param.end_epsilon, epsilon - (-learning_param.end_epsilon +learning_param.init_epsilon) / max_num_iter);
            temperature = max(learning_param.end_temperature,
                              temperature - (-learning_param.end_temperature +learning_param.init_temperature) / max_num_iter);

            //at_walking_solution = boundary.query();

            NeuralNetworkAndScore try_solution = at_walking_solution;

            reptile_step_vary_epsilon(try_solution, f_data, k_iter, local_policy, epsilon,
                                     learning_param.choosePriorityTrain);

            //at_walking_solution.printWeights();

            NeuralNetworkAndScore try_score = evaluate_learner(try_solution, f_data, print, eval_param,
                                                      learning_param.choosePriorityTrain);

            local_policy.update(&try_score);

            bool pushed_boundary = boundary.update(try_score);

            if (pushed_boundary || try_score < at_walking_solution) {
                at_walking_solution = try_score;

                if (at_walking_solution < global_best_solution) {
                    global_best_solution = at_walking_solution;
                }

                if(pushed_boundary && print) {
                    print_stats(epsilon, temperature, iter, k_iter, at_walking_solution, global_best_solution,
                                boundary, local_policy, f_data);
                }

            }
            else if (explore_step(at_walking_solution, try_score, temperature, print)) {
                at_walking_solution = try_score;
                local_policy.update(&at_walking_solution);
            }
            else {
                if(print)
                    cout << "attempts reset ";
                if (100 * temperature < rand(0, 100)) {
                    if(print)
                        cout << "RESETS" <<endl;
                    at_walking_solution = boundary.query();
                }
                else
                {
                    if(print)
                        cout << endl;
                }

            }

            local_policy.update(&at_walking_solution);
        }

        return global_best_solution;
    }


    void update_net_and_score(NeuralNetworkAndScore &gets, NeuralNetworkAndScore sends)
    {
        gets = sends;
    }

    void print_ordering(Policy local_policy, vector<datatype> f_data, vector<BackpropTrajectory> tarjans)
    {
        assert(local_policy.orderings_set_up);
        for(int i = 0; i<local_policy.observed_ordered.size();i++)
        {
            cout << fixed << setprecision(4)
            << f_data[local_policy.observed_ordered[i].second].printConcatinateOutput()
            <<"\t"<< f_data[local_policy.observed_ordered[i].second].score.print()
            << "\t"<< local_policy.observed_ordered[i].first
            << "\t" << tarjans[i].print() << endl;
        }
        cout << "#swaps = " << local_policy.bubble_sort_count() << endl;
    }

    void print_stats(double epsilon, double temperature, int iter, int k_iter,
            NeuralNetworkAndScore at_walking_solution, NeuralNetworkAndScore global_best_solution,
            non_dominating_score_boundary boundary,
            Policy local_policy, vector<datatype> f_data)
    {


        cout << "------------------------------------------------------------------" <<endl;

        print_ordering(local_policy, f_data, at_walking_solution.tarjans);

        cout << endl;

        cout << "iter        = " << iter << endl;
        cout << "epsilon     = " << epsilon <<endl;
        cout << "k_iter      = " << k_iter << endl;
        cout << "at_solution = " << at_walking_solution.print() << endl;
        cout << "global_best = " << global_best_solution.print() << endl;

        cout << endl;

        boundary.print();

        cout << endl;

        cout << "------------------------------------------------------------------" <<endl;
        tout << at_walking_solution.clean_print() << endl;
    }


    void print_stats(
            double epsilon,
            int count_stagnation,
            int global_stagnation,
            int iter,
            int k_iter,
            int k_iter_loops,
            NeuralNetworkAndScore at_walking_solution,
            NeuralNetworkAndScore SA_iter_best_solution,
            NeuralNetworkAndScore best_solution,
            non_dominating_score_boundary boundary,
            Policy local_policy,
            vector<datatype> f_data)
    {

        cout << "------------------------------------------------------------------" <<endl;

        print_ordering(local_policy, f_data, at_walking_solution.tarjans);

        cout << endl;

        cout << "epsilon = " << epsilon <<endl;
        cout << "local stangation = " << count_stagnation << endl;
        cout << "global stangation = " << global_stagnation << endl;
        cout << "k_iter = " << k_iter << endl;
        cout << "k_iter_loops = " << k_iter_loops << endl;
        cout << iter << " at            = " << at_walking_solution.print() << endl;
        cout << iter << " SA local best = " << SA_iter_best_solution.print() << endl;
        cout << iter << " reptile  best = " << best_solution.print() << endl;

        cout << endl;

        boundary.print();


        cout << endl;

        cout << "------------------------------------------------------------------" <<endl;
        tout << at_walking_solution.clean_print() << endl;
    }


    void reptile_train(
            NeuralNetwork &global_best_solution,
            vector<Data> data,
            int root_iter_cutoff,
            evaluate_learner_parameters learning_param)
    {
        //cout << "in reptile_train" <<endl;
        for(int i = 0; i < root_iter_cutoff; i++)
        {
            reptile_step(global_best_solution, data, learning_param.iter_cutoff, learning_param.threshold);
        }
    }


    void reptile_step(
            NeuralNetwork &try_learner,
            vector<datatype> data,
            int iter_cutoff,
            int i,
            double learning_rate,
            double epsilon,
            NeuralNetwork::PriorityTrainReport (NeuralNetwork::*choosePriorityTrain)(Data*, NeuralNetwork::parameters))
    {

        NeuralNetwork train_net = NeuralNetwork(try_learner.copy());
        NeuralNetwork control_net = NeuralNetwork(try_learner.copy());

        NeuralNetwork::parameters train_param;
        NeuralNetwork::parameters control_param;

        if(learning_rate < 0)
        {
            train_param = choose_learning_rate_cutoff_param(iter_cutoff, 0.01, learning_rate);
            control_param = choose_learning_rate_cutoff_param(iter_cutoff, 0.01, -learning_rate);
        }
        else {
            train_param = choose_learning_rate_cutoff_param(iter_cutoff, 0.01, learning_rate);
            control_param = choose_learning_rate_cutoff_param(iter_cutoff, 0.01, learning_rate);
        }


        NeuralNetwork::PriorityTrainReport train_num_iter = train_net.train(&data[i], train_param, choosePriorityTrain);

        double train_error = train_net.get_error_of_data(&data[i]).second;

        NeuralNetwork::PriorityTrainReport control_num_iter = control_net.train(&data[i], control_param, choosePriorityTrain);

        double control_error = control_net.get_error_of_data(&data[i]).second;


        train_net.minus(try_learner);
        //cout << "epsilon = " << epsilon <<endl;
        train_net.mul(-epsilon);
        try_learner.minus(train_net);


        NeuralNetwork new_learner = NeuralNetwork(try_learner.copy());

        NeuralNetwork::PriorityTrainReport new_num_iter = new_learner.train(&data[i], control_param, choosePriorityTrain);

        double new_error = new_learner.get_error_of_data(&data[i]).second;

        //cout << control_error << " -> " << new_error <<endl;
    }


    void reptile_step(
            NeuralNetwork &try_learner,
            vector<datatype> data,
            int iter_cutoff,
            int i)
    {
        int learning_rate = 1;
        reptile_step(try_learner, data, iter_cutoff, i, learning_rate, 0.33, &NeuralNetwork::softPriorityTrain);
    }


    void reptile_step(
            NeuralNetwork &try_learner,
            vector<datatype> data,
            int iter_cutoff,
            int i,
            double epsilon)
    {
        int learning_rate = 1;
        reptile_step(try_learner, data, iter_cutoff, i, learning_rate, epsilon, &NeuralNetwork::softPriorityTrain);
    }


    void reptile_step(
            NeuralNetwork &try_learner,
            vector<datatype> data,
            int iter_cutoff)
    {
        int i = rand(0, data.size()-1);
        reptile_step(try_learner, data, iter_cutoff, i, 0.33);
    }

    void reptile_step_vary_epsilon(
            NeuralNetwork &try_learner,
            vector<datatype> data,
            int iter_cutoff,
            Policy local_policy,
            double epsilon,
            NeuralNetwork::PriorityTrainReport (NeuralNetwork::*choosePriorityTrain)(Data*, NeuralNetwork::parameters))
    {
        //for(int i = 0;i<data.size();i++)
        pair<int, double> policy_decision = local_policy.query();
        assert(policy_decision.second != 0);
        //cout << policy_decision.f <<" "<< policy_decision.second <<endl;
        reptile_step(try_learner, data, iter_cutoff, policy_decision.first, policy_decision.second, epsilon, choosePriorityTrain);
    }

    void reptile_step(
            NeuralNetwork &try_learner,
            vector<datatype> data,
            int iter_cutoff,
            Policy local_policy)
    {
        //for(int i = 0;i<data.size();i++)
        pair<int, double> policy_decision = local_policy.query();
        assert(policy_decision.second != 0);
        reptile_step(try_learner, data, iter_cutoff, policy_decision.first, policy_decision.second, 0.33, &NeuralNetwork::softPriorityTrain);
    }

    void evaluate_unit_task_print(DataAndDecisionTreeScore* data, int local_iter, double local_error)
    {
        cout << data->print(0) <<"\t"<< local_iter << "\t" << local_error << endl;
    }

    void evaluate_unit_task_print(Data* data, int local_iter, double local_error)
    {
        cout << data->printConcatinateOutput() <<"\t"<< local_iter << "\t" << local_error << endl;
    }


    NeuralNetworkAndScore evaluate_unit_task(NeuralNetwork try_learner, datatype* data, bool print, evaluate_learner_parameters learning_param,
            NeuralNetworkAndScore &score, NeuralNetwork::PriorityTrainReport (NeuralNetwork::*choosePriorityTrain)(Data*, NeuralNetwork::parameters))
    {
        //cout << i <<endl;
        NeuralNetwork local_try_learner = NeuralNetwork(try_learner.copy());
        NeuralNetwork::parameters train_param = cutoff_param(learning_param.iter_cutoff, 0.01);
        NeuralNetwork::PriorityTrainReport tarjan = local_try_learner.train(data, train_param, choosePriorityTrain);

        /*
        NeuralNetwork exstended_try_learner = NeuralNetwork(try_learner.copy());
        NeuralNetwork::parameters exstended_cuttoff_param = cutoff_param(learning_param.exstended_iter_cutoff, 0.01);
        NeuralNetwork::PriorityTrainReport exstended_tarjan = exstended_try_learner.train(data, exstended_cuttoff_param, choosePriorityTrain);
        */

        int local_iter = tarjan.total_iter;
        
        NeuralNetworkAndScore individual_score;

        individual_score.tarjan = tarjan;

        individual_score.is_init_score = false;
        score.is_init_score = false;

        pair<double, double> local_error = local_try_learner.get_error_of_data(data);
        double sum_error_over_rows = local_error.first;
        double max_error_over_rows = local_error.second;

        score.max_error = max(score.max_error, max_error_over_rows);
        score.sum_of_max_errors += max_error_over_rows;
        score.max_of_sum_errors = max(score.max_of_sum_errors, sum_error_over_rows);
        score.sum_error += sum_error_over_rows;

        score.update_max_leaf_iter(local_iter);
        individual_score.max_error = max_error_over_rows;
        individual_score.sum_error = sum_error_over_rows;

        if(local_iter == learning_param.iter_cutoff && local_error.second > learning_param.threshold)
        {
            score.num_train_fail++;
            individual_score.num_train_fail = 1;
            //break;
        }
        else
        {
            individual_score.max_train_iter = local_iter;
            score.max_train_iter = max(score.max_train_iter, local_iter);
            score.sum_train_iter+=local_iter;
        }

        if(print)
        {
//            assert(0); // need to implement better local_iter print function in evaluate_unit_task_print
            evaluate_unit_task_print(data, local_iter, local_error.second);
        }

        return individual_score;
    }

    NeuralNetworkAndScore evaluate_learner(
            NeuralNetwork try_learner,
            vector<datatype> data,
            bool print,
            evaluate_learner_parameters learning_param,
            NeuralNetwork::PriorityTrainReport (NeuralNetwork::*choosePriorityTrain)(Data*, NeuralNetwork::parameters))
    {

        vector<NeuralNetworkAndScore> individual_scores;

        NeuralNetworkAndScore score = try_learner;

        score.clear_vals();

        for(int i = 0;i<data.size();i++)
        {
            individual_scores.push_back
            (evaluate_unit_task(try_learner, &data[i], print, learning_param, score, choosePriorityTrain));
        }

        score.set_individual_scores(individual_scores);

        return score;
    }

    NeuralNetworkAndScore evaluate_learner_softPriorityTrain(
            NeuralNetwork try_learner,
            vector<datatype> data,
            bool print,
            evaluate_learner_parameters learning_param)
    {
        return evaluate_learner(try_learner, data, print, learning_param, &NeuralNetwork::softPriorityTrain);
    }


    void meta_train(NeuralNetworkAndScore learner, vector<datatype> data, double treshold)
    {

        int k_iter = 30;
        //performs meta training, using the max error after k iter of a task as signal.
        reptile_SA_train(learner, data, k_iter, treshold, true);
        evaluate_learner(learner, data, true, k_iter, treshold);
    }

    /*NeuralNetwork train_on_local_family()
    {
        bool prev_print_close_local_data_model = print_close_local_data_model;
        print_close_local_data_model = false;
        int init_F_size = (int)data_of_functions.size();
        for(int i = 0;i<init_F_size;i++)
        {
            for(int j = 0; j<data_of_functions[i].numOutputs; j++)
            {
                for(int k = 0;k<data_of_functions[i].size();k++)
                {
                    Datatype new_in_F = data_of_functions[i];
                    new_in_F.out[k][j].flip_value();
                    data_of_functions.push_back(new_in_F);
                }
            }
        }

        train_old();

        print_close_local_data_model = prev_print_close_local_data_model;

        return NeuralNetwork(learner);
    }*/

    void train_old(NeuralNetwork &the_learner, vector<datatype> data, double treshold)
    {
        NeuralNetworkAndScore init_score = the_learner;
        init_score.is_init_score = true;
        train_SA(init_score, data, 800, treshold);
        the_learner = init_score;
    }

    bool pass(NeuralNetworkAndScore prev_energy, NeuralNetworkAndScore new_energy, double temperature, NeuralNetworkAndScore best)
    {
        assert(prev_energy.has_value());
        if(!new_energy.has_value())//failed test
        {
            cout << "fail" <<endl;
            return false;
        }
        //cout << "compare = " << new_energy <<" "<< prev_energy << endl;
        if(new_energy < prev_energy)
        {
            cout << "take" <<endl;
            return true;
        }
        else
        {
            double acc = 0.15*temperature*(prev_energy*0.8+best*0.2)/new_energy;//exp(-(new_energy/prev_energy)/(temperature));
            //cout << "e, e', acc, temp = " << prev_energy << " "<< new_energy <<" " << acc << " " << temperature << endl;
            bool here =  acc > (double)rand(0, 1000)/1000;
            //assert(!here);
            //cout << "here = " << here <<endl;
            return here;
        }
    }

    void train_SA(NeuralNetwork &best_soution, vector<datatype> data, int max_iter, double treshold)
    {

        int num_SA_steps = 40;
        NeuralNetworkAndScore at = best_soution;
        NeuralNetworkAndScore prev_solution;
        do
        {
            cout << "SA WITH ITER = " << num_SA_steps << endl;
            prev_solution = at;
            //at = best_soution;
            double init_diameter = 0.6;
            double min_diameter = 0.001;
            simulated_annealing(at, data, max_iter, num_SA_steps, init_diameter, min_diameter, 0, treshold);
            num_SA_steps*=2;
            cout << "compare = " << at.print() <<" "<< prev_solution.print() <<endl;
        }while(at < prev_solution && num_SA_steps <= 160);
        best_soution = prev_solution;
    }

    void simulated_annealing(
                NeuralNetworkAndScore &best_solution,
                vector<datatype> data,
                int max_iter,
                int num_SA_steps,
                double init_diameter,
                double min_diameter,
                int depth_left,
                double treshold)
    {
        best_solution = evaluate_learner(best_solution, data, false, 1000, treshold);

        double at_diameter = init_diameter;
        NeuralNetworkAndScore at_walking_solution = best_solution;

        for(int iter = 1; iter < num_SA_steps; iter++)
        {
            at_diameter = init_diameter - iter*(init_diameter-min_diameter)/num_SA_steps;

            if(depth_left == 1)cout << "at_D = " << at_diameter <<endl;

            NeuralNetwork next_step = step(at_walking_solution, at_diameter);

            //int old_iter_cutoff = at_walking_solution.max_train_iter*2+(!at_walking_solution.has_value())*treshold;


            NeuralNetworkAndScore next_step_evaluation =
                    evaluate_learner(next_step, data, false, max_iter, treshold);

            //cout << "consider = " << next_step_evaluation << endl;

            if(false && depth_left)
            {
                assert(0);
                simulated_annealing(next_step_evaluation, data, max_iter, 6, at_diameter/2, min_diameter/10, depth_left-1, treshold);
                next_step = next_step_evaluation;
                next_step_evaluation = evaluate_learner(next_step, data, false, max_iter, treshold);
            }

            //next_step_evaluation = local_entropy_learner_evaluation(local_local_learner, data, false, 800);

            if(pass(at_walking_solution, next_step_evaluation, (double)(num_SA_steps-iter)/num_SA_steps, best_solution))
            {
                //cout << "HERE" <<endl;
                at_walking_solution = next_step_evaluation;
                //cout << "at_Walking solution = " << at_walking_solution <<endl;
                if(at_walking_solution<best_solution)
                {
                    //cout << "ULTIMATE HERE" <<endl;
                    best_solution = at_walking_solution;

                    //local_learner.printWeights();
                }
            }
            else
            {

            }
            if(depth_left == 1)
            {
                cout << iter << " at   =  " << at_walking_solution.print() <<endl;
                cout << iter << " best = " << best_solution.print() <<endl;
            }
            /*if(best_solution.max_train_iter <= 1)
            {
                break;
            }*/
        }
    }

    /*void expand_iterative_data(vector<Datatype> &iterative_data, int batch_size)
    {
        for(int i = (int)iterative_data.size(), init_i = i; iterative_data.size()<init_i+batch_size && i < data_of_functions.size(); i++)
        {
            iterative_data.push_back(data_of_functions[i]);
        }
    }*/

    bool learner_treshold_test_on_iterative_sapce(NeuralNetwork learner, vector<datatype> iterative_data, int max_cutoff, double treshold)
    {
        NeuralNetworkAndScore num_iter = evaluate_learner(learner, iterative_data, false, max_cutoff, treshold);
        return num_iter.has_value();
    }

    /*void progressive_train()
    {
        learner = NeuralNetwork(n, n, 1);
        //learner.set_special_weights();

        vector<Datatype> iterative_data;

        int threshold = 1;
        for(int i = 0, batch_size = 1; i<data_of_functions.size(); i += batch_size)
        {
            bool succeed = false;
            vector<Datatype> local_iterative_data;

            while(!succeed)
            {
                local_iterative_data.clear();
                vector<NeuralNetworkAndScore> individual_scores;
                evaluate_learner(learner, data_of_functions, false, threshold, individual_scores);
                for(int i = 0;i<individual_scores.size();i++)
                {
                    if(individual_scores[i].has_value())
                    {
                        local_iterative_data.push_back(data_of_functions[i]);
                    }
                }
                //succeed = (local_iterative_data.size() > 2*iterative_data.size() || local_iterative_data.size() == data_of_functions.size());
                succeed = (local_iterative_data.size() > iterative_data.size());
                threshold*=2;
            }



            batch_size = local_iterative_data.size() - iterative_data.size();
            iterative_data = local_iterative_data;
            double rez = evaluate_learner(learner, local_iterative_data, true);
            cout << "max_train_iter = " << rez <<endl;
            double train_threshold = max(300.0, rez);

            //cout << "train threshold = " << train_threshold << endl;

            train(learner, iterative_data, train_threshold);

            threshold = (evaluate_learner(learner, iterative_data, false, learner.max_train_iter+1).max_train_iter+1)*2.5;
            assert(threshold!=-1 && threshold >= 2);
            cout << "i, iterative_data_size = " << i+batch_size <<" "<< iterative_data.size() <<endl;
        }

    }
    */

    int local_entropy_learner_evaluation(NeuralNetwork try_learner, vector<datatype> data, bool print, int iter_cutoff, double treshold)
    {
        int c = 10;
        int rez = 0;
        bool fail = false;
        for(int i = 0; i<c; i++)
        {
            NeuralNetwork local_step_learner = step(try_learner, 0.05);
            int local_rez =  evaluate_learner(local_step_learner, data, false, 800, treshold);
            rez = max(rez, local_rez);
            if(local_rez == -1)
            {
                fail = true;
            }
            cout << local_rez <<endl;
        }
        if(fail)
        {
            return -1;
        }
        return rez;
    }
};

#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_FIRSTORDERLEARNING_H
