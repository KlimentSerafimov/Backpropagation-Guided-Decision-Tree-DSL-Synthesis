//
// Created by Kliment Serafimov on 2019-04-29.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_NET_AND_SCORE_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_NET_AND_SCORE_H

#include "Header.h"
#include "NeuralNetwork.h"

class NeuralNetworkAndScore: public NeuralNetwork
{
public:


    class four_neural_errors
    {

    public:

        double max_error = 0;
        double sum_error = 0;

        double max_of_sum_errors = 0;
        double sum_of_max_errors = 0;

        four_neural_errors()
        {}
    };

    double max_error = 0;
    double sum_error = 0;

    double max_of_sum_errors = 0;
    double sum_of_max_errors = 0;



    double sum_segments;

    double one_segment;

    double sum_to_one_ratio;

    //vector<four_neural_errors> neural_errors;

    bool is_init_score = true;

    int num_train_fail = 0;

    int max_train_iter = 0;
    int sum_train_iter = 0;

    int max_leaf_iter = (1<<30);

    vector<double> individual_max_errors;

    BackpropTrajectory tarjan;

    vector<BackpropTrajectory> tarjans;

    int max_unit_ordering_error = (1<<30);

    int sum_ordering_error = (1<<30);

    double to_double()
    {
        return max_error;
    }

    /*void printTarjans()
    {
        cout << "Tarjans:" <<endl;
        for(int i = 0;i<tarjans.size();i++)
        {
            cout << tarjan[i].print() <<endl;
        }
        cout << endl;
    }*/

    operator double() const
    {
        return max_error;
    }

    void set_individual_scores(vector<NeuralNetworkAndScore> individual_scores)
    {
        assert(individual_max_errors.size() == 0);
        for(int i = 0;i<individual_scores.size();i++)
        {
            individual_max_errors.push_back(individual_scores[i].max_error);
            tarjans.push_back(individual_scores[i].tarjan);
        }
    }

    NeuralNetworkAndScore()
    {

    }


    NeuralNetworkAndScore(NeuralNetwork self): NeuralNetwork(self)
    {

    }

    void clear_vals()
    {
        //neural_errors.clear();

        is_init_score = true;

        num_train_fail = 0;

        max_train_iter = 0;
        sum_train_iter = 0;

        max_leaf_iter = (1<<30);

        individual_max_errors.clear();

        tarjans.clear();

        tarjan = BackpropTrajectory();

        max_unit_ordering_error = (1<<30);

        sum_ordering_error = (1<<30);
    }

    void update_max_leaf_iter(int new_iter)
    {
        if(max_leaf_iter == (1<<30))
        {
            max_leaf_iter = new_iter;
        }
        else{
            max_leaf_iter = max(max_leaf_iter, new_iter);
        }
    }

    bool operator < (NeuralNetworkAndScore &other) const
    {
        if(is_init_score)
        {
            return false;
        }

        if(other.is_init_score)
        {
            return true;
        }


        /*if(max_error >= 0.4 || other.max_error >= 0.4)
        {
            return max_error < other.max_error;
        }
        else
        */


        //assert(neural_errors.size() == 1);
        //assert(other.neural_errors.size() == 1);

        if(sum_ordering_error == other.sum_ordering_error)
        {

            if(max_unit_ordering_error == other.max_unit_ordering_error)
            {
                return max_error < other.max_error;
                //return neural_errors[0].max_error < other.neural_errors[0].max_error;
            }
            else
            {
                return max_unit_ordering_error < other.max_unit_ordering_error;
            }
        }
        else
        {
            return sum_ordering_error < other.sum_ordering_error;
        }

        //return (max_error < other.max_error || max_leaf_iter < other.max_leaf_iter || ordering_error < other.ordering_error);
    }

    bool has_value()
    {
        return true;
    }

    string print()
    {
        return "sum_error = \t" + std::to_string(sum_error) + "\t max_error = \t" + std::to_string(max_error) +
        "\t max_unit_ord_error = " + std::to_string(max_unit_ordering_error) + "\t sum_ord_error = "  + std::to_string(sum_ordering_error); // + "\t sum_error = \t" + std::to_string(sum_error);
    }

    string clean_print()
    {
        return std::to_string(max_error);// + "\t\t" + std::to_string(sum_error);
    }
};

class non_dominating_score_boundary
{
    vector<NeuralNetworkAndScore> boundary;

    bool is_max_boundary_size_set = false;
    int max_boundary_size;

public:

    void set_max_boundary_size(int _max_boundary_size)
    {
        is_max_boundary_size_set = true;
        max_boundary_size = _max_boundary_size;
    }

    bool update(NeuralNetworkAndScore to_insert)
    {

        vector<NeuralNetworkAndScore> new_boundary;
        for(int i = 0; i < boundary.size();i++)
        {
            if(
                to_insert.max_error >= boundary[i].max_error &&
                to_insert.sum_error >= boundary[i].sum_error &&
                to_insert.max_unit_ordering_error >= boundary[i].max_unit_ordering_error &&
                to_insert.sum_ordering_error >= boundary[i].sum_ordering_error
            )
            {
                //cout << "to_insert is dominated" <<endl;
                return false;
            }

            if(
                to_insert.max_error <= boundary[i].max_error &&
                to_insert.sum_error <= boundary[i].sum_error &&
                to_insert.max_unit_ordering_error <= boundary[i].max_unit_ordering_error &&
                to_insert.sum_ordering_error <= boundary[i].sum_ordering_error
            )
            {
                //cout << "to_insert dominates boundary[" << i <<"]" <<endl;
                //skip, bc boundary[i] is dominated
            }
            else
            {
                new_boundary.push_back(boundary[i]);
            }
        }


        //cout << "to_insert is non-dominated" <<endl;

        new_boundary.push_back(to_insert);

        boundary = new_boundary;

        assert(is_max_boundary_size_set);
        while(boundary.size() > max_boundary_size)
        {

            vector<pair<double, int> > best;
            for(int i = 0;i<boundary.size();i++)
            {
                vector<double> values;
                values.push_back(boundary[i].max_error);
                values.push_back(boundary[i].sum_error);
                values.push_back(boundary[i].max_unit_ordering_error);
                values.push_back(boundary[i].sum_ordering_error);
                if(i == 0) {
                    for (int j = 0; j < values.size(); j++) {
                        best.push_back(make_pair(values[j], i));
                    }
                }
                else
                {
                    for (int j = 0; j < values.size(); j++) {
                        best[j] = min(best[j], make_pair(values[j], i));
                    }
                }
            }

            pair<pair<int, int>, int> worst = make_pair(make_pair(-1, -1), 0);
            for(int i = 0;i<boundary.size();i++)
            {
                bool in_best = false;
                for(int j = 0;j<best.size() && !in_best;j++)
                {
                    if(i == best[j].second)
                    {
                        in_best = true;
                    }
                }
                if(!in_best) {
                    worst = max(worst, make_pair(make_pair(boundary[i].sum_ordering_error, boundary[i].max_unit_ordering_error), i));
                }
            }
            if(worst.first.first != -1) {
                boundary.erase(boundary.begin() + worst.second);
            } else{
                break;
            }
        }

        return true;
    }

    NeuralNetworkAndScore query()
    {
        return boundary[rand(0, boundary.size()-1)];
    }

    void print()
    {
        cout << "-----------------------------------" << endl;
        cout << "BOUNDARY:" <<endl;
        for(int i = 0; i < boundary.size();i++)
        {
            cout << boundary[i].print() <<endl;
        }
        cout << "-----------------------------------" << endl;
    }
};

class meta_net_and_score: public NeuralNetworkAndScore
{
public:

    meta_net_and_score(NeuralNetworkAndScore self): NeuralNetworkAndScore(self)
    {

    }

    meta_net_and_score()
    {

    }

};

#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_NET_AND_SCORE_H
