//
// Created by Kliment Serafimov on 2019-05-21.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_POLICY_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_POLICY_H

#include "Header.h"
#include "Data.h"
#include "NeuralNetworkAndScore.h"
#include "DataAndDecisionTreeScore.h"

class Policy
{
public:

    enum policyType : int {total_drift, controled_drift};

    policyType policy_type;

    int n;

    vector<DecisionTreeScore> desired;

    vector<double> observed;

    bool print;

    Policy(vector<DataAndDecisionTreeScore> f_data)
    {
        print = false;
        policy_type = total_drift;

        n = f_data.size();
        for(int i = 0;i<f_data.size();i++)
        {
            desired.push_back(f_data[i].score);
        }
    }

    /*
    Policy(vector<Data> f_data, bool _print)
    {
        print = _print;
        policy_type = total_drift;
        n = f_data.size();
    }

    Policy(vector<DataAndDecisionTreeScore> f_data, bool _print)
    {
        print = _print;
        policy_type = controled_drift;
        for(int i = 0;i<f_data.size();i++)
        {
            desired.push_back(f_data[i].score);
        }
    }*/


    vector<int> id_to_reduce_error;
    vector<int> id_to_increase_error;

    vector<int> weighed_id_to_reduce_error;
    vector<int> weighed_id_to_increase_error;


    pair<int, pair<int, double> > most_distant;

    pair<int, pair<int, double> > most_distance_to_increase;
    pair<int, pair<int, double> > most_distance_to_reduce;

    pair<int, double> highest_error;


    vector<pair<DecisionTreeScore, int> > desired_ordered;

    vector<pair<double, int> > observed_ordered;

    vector<int> correct_ids;

    vector<int> count_per_id;

    bool orderings_set_up = false;

    int bubble_sort_count()
    {
        vector<pair<DecisionTreeScore, int> > for_bubble_sort;

        for(int i = 0;i<observed_ordered.size();i++) {
            for_bubble_sort.push_back(make_pair(desired[observed_ordered[i].second], observed_ordered[i].second));
        }

        count_per_id = vector<int>(desired.size(), 0);

        int count = 0;
        for(int i = 0; i<for_bubble_sort.size();i++)
        {
            for(int j = for_bubble_sort.size()-2;j>=i;j--)
            {
                if(for_bubble_sort[j+1].first < for_bubble_sort[j].first)
                {
                    assert(for_bubble_sort[j].second < count_per_id.size());
                    assert(for_bubble_sort[j+1].second < count_per_id.size());
                    count_per_id[for_bubble_sort[j].second]++;
                    count_per_id[for_bubble_sort[j+1].second]--;
                    swap(for_bubble_sort[j], for_bubble_sort[j+1]);
                    count++;
                }
            }
        }

        return count;
    }

    void set_up_orderings(NeuralNetworkAndScore* generator)
    {
        observed = generator->individual_max_errors;

        desired_ordered.clear();

        observed_ordered.clear();

        assert(desired.size() == observed.size());

        for (int i = 0; i < desired.size(); i++) {
            desired_ordered.push_back(make_pair(desired[i], i));
            observed_ordered.push_back(make_pair(observed[i], i));
        }

        sort_v(desired_ordered);
        sort_v(observed_ordered);

        orderings_set_up = true;
    }

    int get_swap_count(NeuralNetworkAndScore* generator)
    {
        set_up_orderings(generator);
        return bubble_sort_count();
    }

    pair<int, pair<int, double> > get_most_distant()
    {

        weighed_id_to_reduce_error.clear();
        weighed_id_to_increase_error.clear();
        correct_ids.clear();

        most_distant = make_pair(-1, make_pair(-1, 0));

        most_distance_to_increase = make_pair(-1, make_pair(-1, 0));
        most_distance_to_reduce = make_pair(-1, make_pair(-1, 0));

        int num_correct = 0;

        for(int i = 0;i<count_per_id.size();i++)
        {
            if(count_per_id[i] > 0)
            {
                most_distant = max(most_distant, make_pair(count_per_id[i], make_pair(i, -1.0)));
                most_distance_to_increase = max(most_distance_to_increase, make_pair(count_per_id[i], make_pair(i, -1.0)));
                for (int j = 0; j < count_per_id[i]*count_per_id[i]; j++) {
                    weighed_id_to_increase_error.push_back(i);
                }
            }
            else if(count_per_id[i] < 0)
            {
                most_distant = max(most_distant, make_pair(-count_per_id[i], make_pair(i, +1.0)));
                most_distance_to_reduce = max(most_distance_to_reduce, make_pair(-count_per_id[i], make_pair(i, +1.0)));
                for (int j = 0; j < -count_per_id[i]*(-count_per_id[i]); j++)
                {
                    weighed_id_to_reduce_error.push_back(i);
                }
            }
            else
            {
                num_correct++;
            }
            if(-2 <= count_per_id[i] && count_per_id[i] <= 0)
            {
                correct_ids.push_back(i);
            }
        }

        assert(num_correct == count_per_id.size() || most_distant.second.second != 0);

        if(most_distant.first == -1)
        {
            most_distant.first = 0;
        }

        return most_distant;
    }

    void update(NeuralNetworkAndScore* generator)
    {
        if(policy_type == total_drift)
        {
            set_up_orderings(generator);

            int count = bubble_sort_count();

            /*generator->sum_segments = get_sum_segments();

            generator->one_segment = get_entire_segment();

            generator->sum_to_one_ratio = sum_segments/one_segment;*/

            assert(get_most_distant() == most_distant);

            generator->max_unit_ordering_error = most_distant.first;
            generator->sum_ordering_error = count;
        }
        else if(policy_type == controled_drift)
        {
            set_up_orderings(generator);

            assert(orderings_set_up);

            id_to_reduce_error.clear();
            id_to_increase_error.clear();

            highest_error = observed_ordered[observed_ordered.size()-1];

            bool mismatch = false;

            assert(desired_ordered.size() >= 1);
            DecisionTreeScore threshold_to_train_under = desired_ordered[0].first;

            if(print)cout << "opt:: ";
            for (int i = 0; i < desired_ordered.size(); i++) {
                if(print)cout << desired_ordered[i].first.print(desired_ordered[i].second) <<" " ;
                if (!mismatch) {
                    if (desired_ordered[i].first < desired[observed_ordered[i].second]) {
                        mismatch = true;
                        threshold_to_train_under = desired[observed_ordered[i].second];
                    } else {
                        //id_to_reduce_error.push_back(observed_ordered[i].second);
                    }
                } else {
                    if (desired[observed_ordered[i].second] < threshold_to_train_under) {
                        id_to_reduce_error.push_back(observed_ordered[i].second);
                    }
                }
            }

            if(print)
            {
                cout << endl;
                cout << "obs:: ";
                for(int i = 0;i<observed_ordered.size();i++) {
                    cout << desired[observed_ordered[i].second].print(observed_ordered[i].second) << " ";
                }
                cout << endl;
            }

            mismatch = false;

            DecisionTreeScore threshold_to_anti_train_over = desired_ordered[desired_ordered.size()-1].first;

            for (int i = desired_ordered.size()-1; i >=0; i--) {
                if (!mismatch) {
                    if (desired[observed_ordered[i].second] < desired_ordered[i].first) {
                        mismatch = true;
                        threshold_to_anti_train_over = desired[observed_ordered[i].second];
                    } else {
                        //id_to_reduce_error.push_back(observed_ordered[i].second);
                    }
                } else {
                    if (threshold_to_anti_train_over < desired[observed_ordered[i].second]) {
                        id_to_increase_error.push_back(observed_ordered[i].second);
                    }
                }
            }

            int count = bubble_sort_count();

            assert(get_most_distant() == most_distant);

            generator->max_unit_ordering_error = most_distant.first;
            generator->sum_ordering_error = count;

            if(print)
            {
                cout << "weighed increase:: ";

                for(int i = 0;i<weighed_id_to_increase_error.size();i++)
                {
                    cout << weighed_id_to_increase_error[i] <<" ";
                }
                cout << endl;
                cout << "weighed reduce  :: ";
                for(int i = 0;i<weighed_id_to_increase_error.size();i++)
                {
                    cout << weighed_id_to_reduce_error[i] <<" ";
                }

                cout << endl;
                cout << "correct ids     ::";
                for(int i = 0;i<correct_ids.size();i++)
                {
                    cout << correct_ids[i] <<" ";
                }
                cout << endl;

                cout << "#sw_per_id:: ";
                for(int i = 0;i<count_per_id.size();i++)
                {
                    cout << count_per_id[i] <<" ";
                }
                cout << endl;
                cout << "sum#swaps:: " << count <<endl;

                cout << endl;
            }

        }
        else
        {
            assert(0);
        }
    }


    pair<int, double> query()
    {


        int id = rand(0, desired.size()-1);

        if(print) {
            cout << "correct_drift heuristic" << endl;
            cout << "[+1" << ", " << desired[id].print(id) << "] ";
        }

        return make_pair(id, 1);


        if(policy_type == total_drift || most_distant.first == -1 || most_distant.first == 0 || rand(0, 100) < 40)
            //|| weighed_id_to_reduce_error.size() == 0)
        {
            //if(most_distance_to_reduce.f != -1 )
            //{
            //    return make_pair(most_distance_to_reduce.second.f, most_distance_to_reduce.second.second);
            //}
            //return make_pair(highest_error.second, 1);
            //return make_pair(rand(0, desired.size()-1), 1);

            if(correct_ids.size() >= 1) {
                int id = correct_ids[rand(0, correct_ids.size() - 1)];

                if(print) {
                    cout << "correct_drift heuristic" << endl;
                    cout << "[+1" << ", " << desired[id].print(id) << "] ";

                }
                return make_pair(id, 1);
            }
            else
            {
                int id = rand(0, desired.size()-1);

                if(print) {
                    cout << "correct_drift heuristic" << endl;
                    cout << "[+1" << ", " << desired[id].print(id) << "] ";
                }
                return make_pair(id, 1);
            }
        }
        else if(policy_type == controled_drift)
        {

            if(most_distance_to_reduce.second.second != 0 && most_distance_to_reduce.first != -1 && rand(0, 100) < 75)
            {

                if (print) {
                    cout << "most_distant_to_reduce heuristic" << endl;
                    cout << "[" << most_distance_to_reduce.second.first << ", "
                         << desired[most_distance_to_reduce.second.first].print(most_distance_to_reduce.second.first) << "] ";
                }

                return make_pair(most_distance_to_reduce.second.first, most_distance_to_reduce.second.second);

            }
            else if(most_distant.second.second != 0) {

                assert(most_distant.second.first != -1);
                if (print) {
                    cout << "most_distant heuristic" << endl;
                    cout << "[" << most_distant.second.second << ", " << desired[most_distant.second.first].print(most_distant.second.first)
                         << "] ";
                }
                return make_pair(most_distant.second.first, most_distant.second.second);

            }
            else
            {
                int id = rand(0, desired.size()-1);

                if(print) {
                    cout << "correct_drift heuristic" << endl;
                    cout << "[+1" << ", " << desired[id].print(id) << "] ";
                }

                return make_pair(id, 1);

            }
            /* WEIGHED
             * if(weighed_id_to_reduce_error.size() >= 1)
            {
                assert(weighed_id_to_increase_error.size() >= 1);

                int sum = weighed_id_to_reduce_error.size() + weighed_id_to_increase_error.size();

                if(rand(0, sum) < weighed_id_to_reduce_error.size())
                {

                    int sample = rand(0, weighed_id_to_reduce_error.size() - 1);
                    cout << "[+1," << desired[weighed_id_to_reduce_error[sample]].print(weighed_id_to_reduce_error[sample]) << "] ";
                    return make_pair(weighed_id_to_reduce_error[sample], 1);
                }
                else
                {

                    int sample = rand(0, weighed_id_to_increase_error.size() - 1);
                    cout << "[-1," << desired[weighed_id_to_increase_error[sample]].print(weighed_id_to_increase_error[sample]) << "] ";
                    return make_pair(weighed_id_to_increase_error[sample], -1);
                }
            }
            else
            {
                assert(0);
            }*/

            /*
             * UNWEIGHED
             * if(id_to_reduce_error.size() >= 1)
            {
                assert(id_to_increase_error.size() >= 1);

                if(rand(0, 100) < 50)
                {

                    int sample = rand(0, id_to_reduce_error.size() - 1);
                    cout << "[+1," << desired[id_to_reduce_error[sample]].print(id_to_reduce_error[sample]) << "] ";
                    return make_pair(id_to_reduce_error[sample], 1);
                }
                else
                {

                    int sample = rand(0, id_to_increase_error.size() - 1);
                    cout << "[-1," << desired[id_to_increase_error[sample]].print(id_to_increase_error[sample]) << "] ";
                    return make_pair(id_to_increase_error[sample], -1);
                }

            }
            else
            {
                assert(id_to_increase_error.size() == 0);
                return make_pair(rand(0, desired.size()-1), 1);
            }*/
        }
        else
        {

            int id = rand(0, desired.size()-1);

            if(print) {
                cout << "correct_drift heuristic" << endl;
                cout << "[+1" << ", " << desired[id].print(id) << "] ";
            }
            return make_pair(id, 1);

        }
    }
};


#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_POLICY_H
