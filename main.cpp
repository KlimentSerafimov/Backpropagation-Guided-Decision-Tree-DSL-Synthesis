
/* Code written by Kliment Serafimov */

#include "neuron.h"
#include "layer.h"
#include "Data.h"
#include "Header.h"
#include "bit_signature.h"
#include "util.h"
#include "layered_graph.h"
#include "batch.h"
#include "net.h"
#include "decision_tree.h"
#include "firstOrderLearning.h"
#include "Policy.h"
#include "dp_decision_tree.h"

#include "archaic_neural_designed_circuit.h"

ofstream fout("table.out");

ofstream fout_experiment("experiment.out");

ofstream fout_dataset("dataset.out");

template<typename datatype>
class firstOrderDataset
{
public:

    vector<datatype> train_data;
    vector<datatype> test_data;

    firstOrderDataset()
    {

    }

    void print()
    {
        cout << "TRAIN" <<endl;
        for(int i = 0;i<train_data.size();i++)
        {
            cout << train_data[i].print() << endl;
        }
        cout << "TEST" <<endl;
        for(int i = 0;i<test_data.size();i++)
        {
            cout << test_data[i].print() << endl;
        }

    }
};

template<typename datatype>
class secondOrderDataset
{
public:
    int n;
    vector<vector<datatype> > train_meta_data;
    vector<vector<datatype> > test_meta_data;


};

class GeneralizeDataset
{
    Data train_data;
    Data generalize_data;//like validation set, but used in training
};

class firstOrderLearnToGeneralize
{
    vector<GeneralizeDataset> learn_to_generalize_data;
    vector<GeneralizeDataset> test_generalization_data;
};

template<typename datatype>
class secondOrderLearning: public firstOrderLearning<datatype>
{
public:

    secondOrderLearning(): firstOrderLearning<datatype>()
    {

    }

    void learn_to_meta_learn(
            meta_net_and_score &learner,
            vector<vector<Data> > train_meta_data,
            int root_iter, typename firstOrderLearning<datatype>::evaluate_learner_parameters params)
    {
        learning_to_reptile(learner, train_meta_data, root_iter, params);
    }

    void learning_to_reptile(
            meta_net_and_score &global_best_solution,
            vector<vector<Data> > f_data,
            int root_iter_init,
            typename firstOrderLearning<datatype>::evaluate_learner_parameters params)
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
                evaluate_meta_learner(global_best_solution, f_data, false, root_iter, params);

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
                at_walking_solution = evaluate_meta_learner(SA_iter_best_solution, f_data, false, root_iter, params);
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
                    net_and_score new_score = at_walking_solution =
                                                      evaluate_meta_learner(at_walking_solution, f_data, true, root_iter, params);

                    //cout << new_score << endl;

                    //at_walking_solution.printWeights();

                    if (new_score < SA_iter_best_solution) {
                        SA_iter_best_solution = new_score;
                        count_stagnation = 0;

                        firstOrderLearning<datatype>::update_solutions
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
                            net next_step = firstOrderLearning<datatype>::step(at_walking_solution, radius);
                            cout << "SA step" << endl;
                            new_score = SA_iter_best_solution = at_walking_solution =
                                    evaluate_meta_learner(next_step, f_data, true, root_iter, params);

                            count_stagnation=0;
                            global_stagnation++;
                            k_iter = leaf_iter_init;


                            firstOrderLearning<datatype>::update_solutions
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
            net &try_learner,
            vector<vector<Data> > data,
            int root_cutoff,
            typename firstOrderLearning<datatype>::evaluate_learner_parameters params)
    {
        int i;

        i = rand(0, data.size()-1);

        {
            //cout << "train on task: " << i <<endl;
            //try_learner.printWeights();
            net local_try_learner = net(try_learner.copy());
            //net_and_score before_train = evaluate_learner(local_try_learner, data[i], print, leaf_cutoff);

            firstOrderLearning<datatype>::reptile_train(local_try_learner, data[i], root_cutoff, params);

            local_try_learner.minus(try_learner);
            local_try_learner.mul(-1.0/data.size());
            try_learner.minus(local_try_learner);
        }

        //cout << "average = " << (double)sum_train_iter/Dataset_of_functions.size() <<endl;
    }

    meta_net_and_score evaluate_meta_learner(
            net try_learner,
            vector<vector<Data> > data,
            bool print,
            int root_iter,
            typename firstOrderLearning<datatype>::evaluate_learner_parameters params)
    {
        meta_net_and_score score = net_and_score(try_learner);

        score.clear_vals();

        if(print)
        {
            cout << "IN evaluate_meta_learner" << endl;
        }
        for(int i = 0;i<data.size();i++)
        {
            //cout << i <<endl;
            net local_try_learner = net(try_learner.copy());


            firstOrderLearning<datatype>::reptile_train(local_try_learner, data[i], root_iter, params);

            net_and_score reptile_score =
                    firstOrderLearning<datatype>::evaluate_learner_softPriorityTrain(local_try_learner, data[i], print, params);

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

class f_and_score: public DecisionTreeScore
{
public:
    long long f;

    f_and_score(int _f, DecisionTreeScore tmp): DecisionTreeScore()
    {
        f = _f;
        size = tmp.size;
        num_solutions = tmp.num_solutions;
    }

    operator int()
    {
        return f;
    }
};

DecisionTreeScore _get_opt_decision_tree_score(int n, int func)
{
    //assert(0);
    cout << "here?" <<endl;
    cout << "buffer 2" <<endl;
    Data local_new_data = Data();
    assert(0);
    local_new_data.init_exaustive_table_with_unary_output(n, func);

    dp_decision_tree<Data> decision_tree_solver;
    DecisionTreeScore size_opt;
    net::parameters cutoff_parametes;
    size_opt = decision_tree_solver.synthesize_decision_tree_and_get_size
            (cutoff_parametes, n, local_new_data, optimal);

    return size_opt;
}

void ttmp(int n, int func, Data local_new_data)
{
    //return size_opt;
}

string get_tab(int num_tabs)
{
    string ret;
    for(int i = 0;i<num_tabs;i++)
    {
        ret+="\t";
    }
    return ret;
}

string print_dt_string(string dt_string,  int num_tabs, char open, char close, bool do_print)
{
    string tabbed_string;
    for(int i = 0;i<dt_string.size();i++)
    {
        tabbed_string += get_tab(num_tabs);
        cout << get_tab(num_tabs);
        while(dt_string[i] != open && dt_string[i] != close && i < dt_string.size())
        {
            tabbed_string += dt_string[i];
            cout << dt_string[i];
            i++;
        }
        if(dt_string[i] == open)
        {
            tabbed_string += "\n";
            cout << "\n";

            if(do_print) {
                tabbed_string += get_tab(num_tabs) + dt_string[i] + "\n";
                cout << get_tab(num_tabs) << dt_string[i] << "\n";
            }
            num_tabs++;
        }
        else if(dt_string[i] == close)
        {
            tabbed_string += "\n";
            cout << "\n";
            num_tabs--;

            if(do_print) {
                tabbed_string += get_tab(num_tabs) + dt_string[i] + "\n";
                cout << get_tab(num_tabs) << dt_string[i] << endl;
            }
        }
    }
    //fout_dataset << tabbed_string << endl;
    return tabbed_string;
}

string print_dt_string_python(string dt_string,  int num_tabs, char open, char close)
{
    string tabbed_string;
    for(int i = 0;i<dt_string.size();i++)
    {
        tabbed_string += get_tab(num_tabs);
        cout << get_tab(num_tabs);
        while(dt_string[i] != open && dt_string[i] != close && i < dt_string.size())
        {
            tabbed_string += dt_string[i];
            cout << dt_string[i];
            i++;
        }
        if(dt_string[i] == open)
        {

            tabbed_string += "\n";
            cout << "\n";
//            tabbed_string += get_tab(num_tabs) + dt_string[i] + "\n";
//            cout << get_tab(num_tabs) << dt_string[i] << "\n";
            num_tabs++;
        }
        else if(dt_string[i] == close)
        {
            tabbed_string += "\n";
            cout << "\n";
            num_tabs--;
//            tabbed_string += get_tab(num_tabs) + dt_string[i] + "\n";
//            cout << get_tab(num_tabs) << dt_string[i] <<endl;
        }
    }
    //fout_dataset << tabbed_string << endl;
    return tabbed_string;
}

string print_f(int n, int f)
{
    string tabbed_string;
    for(int i = 0; i <(1<<n);i++)
    {
        tabbed_string += toBinaryString(i, n) + "\n" + to_string(((f & (1<<i))  != 0)) + "\n";
        cout << toBinaryString(i, n) <<" " << ((f & (1<<i))  != 0) <<endl;
        //cout << toBinaryString(f, (1<<n)) <<endl;
    }
    return tabbed_string;
}

string print_decision_tree_of_f_cpp(int n, int func)
{
    cout << "here in print_decision_tree_of_f_cpp" <<endl;
    DecisionTreeScore score = _get_opt_decision_tree_score(n, func);
    string tabbed_string;
    assert(score.dt_strings.size() == score.if_cpp_format_strings.size());
    for(int i = 0;i<score.dt_strings.size();i++) {
        string local_str = score.if_cpp_format_strings[i];//"(dt=" + score.dt_strings[i] + ")";
        tabbed_string+=print_dt_string(local_str, 0, '{', '}', true);
        tabbed_string+="\n";
        cout << "\n";
    }
    tabbed_string+="\n";
    cout << "\n";
    //fout_dataset << tabbed_string << endl;
    return tabbed_string;
}

//string print_decision_tree_of_f_python(int n, int func)
//{
//    string tabbed_string;
//    cout << "here in print_decision_tree_of_f_python" <<endl;
//    DecisionTreeScore score = _get_opt_decision_tree_score(n, func);
//    assert(score.dt_strings.size() == score.if_cpp_format_strings.size());
//    for(int i = 0;i<score.dt_strings.size();i++) {
//        string local_str = score.if_python_format_strings[i];//"(dt=" + score.dt_strings[i] + ")";
//        tabbed_string+=print_dt_string_python(local_str, 0, '{', '}');
//        tabbed_string+="\n";
//        cout << "\n";
//    }
//    tabbed_string+="\n";
//    cout << "\n";
//    //fout_dataset << tabbed_string << endl;
//    return tabbed_string;
//}

vector<f_and_score> get_smallest_f(int n)
{

    assert(n <= 3);

    vector<int> fs;

    for (int i = 0; i < (1 << (1 << n)); i++) {
        fs.pb(i);
    }

    vector<f_and_score> ordering;

    for(int i = 0;i<fs.size();i++)
    {
        cout << "here in get_smallest_f" <<endl;
        ordering.pb(f_and_score(fs[i], _get_opt_decision_tree_score(n, fs[i])));
    }

    //sort_v(ordering);

    return ordering;

}



void print_all_dt_strings(int n)
{
    string tabbed_string = "";
    assert(n <= 4);
    for(int i = 0; i < (1<<(1<<n)); i++)
    {
        tabbed_string += "File file = File(name=\"out_to_str " + toBinaryString(i, (1<<n)) + "\",";
        tabbed_string += "dataset=\"";
        //tabbed_string += print_f(n, i);
        tabbed_string += "\")\n";
        tabbed_string += "File file = File(name=\"dt_of_f " + toBinaryString(i, (1<<n)) + "\",";
        tabbed_string += "dataset=\"";

        int func = i;
        cout << "here in print_decision_tree_of_f_python" <<endl;

        Data local_new_data = Data();
        local_new_data.init_exaustive_table_with_unary_output(n, func);

        dp_decision_tree<Data> decision_tree_solver = dp_decision_tree<Data>();
        DecisionTreeScore size_opt;
        net::parameters cutoff_parametes;
        size_opt = decision_tree_solver.synthesize_decision_tree_and_get_size
                (cutoff_parametes, n, local_new_data, optimal);

        DecisionTreeScore score = size_opt;

        string local_tabbed_string;
        assert(score.dt_strings.size() == score.if_python_format_strings.size());
        for(int i = 0;i<score.dt_strings.size();i++) {
            string local_str = score.if_python_format_strings[i];//"(dt=" + score.dt_strings[i] + ")";
            local_tabbed_string+=print_dt_string(local_str, 0, '{', '}', false);
            local_tabbed_string+="\n";
            cout << "\n";
        }

        local_tabbed_string+="\n";
        cout << "\n";

        tabbed_string += local_tabbed_string + "\")\n";
        tabbed_string += "-------------------";
        cout << "-------------------" << endl;
    }
    //fout_dataset << tabbed_string << endl;
}

firstOrderDataset<DataAndScore> init_random_test_set(int n)
{
    firstOrderDataset<DataAndScore> ret;
    int test_num_samples = 64;

    vector<f_and_score> sample_ids;

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
        sample_ids.pb(f_and_score(sample_id, _get_opt_decision_tree_score(n, sample_id)));
    }

    //test

    for(int i = 0;i<test_num_samples;i++)
    {
        DataAndScore new_data(sample_ids[i]);
        new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].f);
        ret.test_data.pb(new_data);
    }

    ret.print();

    return ret;
}

firstOrderDataset<DataAndScore> init_custom_data_and_difficulty_set(int n, double training_set_percent, double testing_set_percent)
{
    assert(n<=3);
    firstOrderDataset<DataAndScore> ret;

    //int train_num_samples; //defined later
    int test_num_samples = (1<<(1<<n));




    vector<f_and_score> sample_ids = get_smallest_f(n);

    if(n == 3)
    {

        srand(0);

        for(int i = 0;i<sample_ids.size();i++)
        {
            if(sample_ids[i].size < 7)
            {
                DataAndScore new_data(sample_ids[i]);
                new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].f);

                if (rand(0, 100) < testing_set_percent) {
                    ret.test_data.pb(new_data);
                } else {
                    if (rand(0, 100) < training_set_percent) {
                        ret.train_data.pb(new_data);
                    }
                }
            }
        }


        if(true) {
            int original_seed = 0;
            for (int i = 0; i < 8; i++) {
                int seed = original_seed;
                if ((seed & (1 << i)) != 0) {
                    seed -= (1 << i);
                } else {
                    seed |= (1 << i);
                }

                DataAndScore new_1_data(sample_ids[seed]);
                {
                    new_1_data.init_exaustive_table_with_unary_output(n, sample_ids[seed].f);

                    if(false) {
                        if (rand(0, 100) < testing_set_percent) {
                            ret.test_data.pb(new_1_data);
                        } else {
                            if (rand(0, 100) < training_set_percent) {
                                ret.train_data.pb(new_1_data);
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
                    DataAndScore new_2_data(sample_ids[seed]);
                    if (new_2_data.score.size >= 9) {
                        new_2_data.init_exaustive_table_with_unary_output(n, sample_ids[seed].f);

                        if(false) {
                            if (rand(0, 100) < testing_set_percent) {
                                ret.test_data.pb(new_2_data);
                            } else {
                                if (rand(0, 100) < training_set_percent) {
                                    ret.train_data.pb(new_2_data);
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
                        DataAndScore new_3_data(sample_ids[seed]);
                        if(sample_ids[seed].size == 7)
                            if (true) {
                                new_3_data.init_exaustive_table_with_unary_output(n, sample_ids[seed].f);

                                if (rand(0, 100) < testing_set_percent) {
                                    ret.test_data.pb(new_3_data);
                                } else {
                                    if (rand(0, 100) < training_set_percent) {
                                        ret.train_data.pb(new_3_data);
                                    }
                                }
                            }
                    }
                }
            }
        }
    }

    //train
    if(n <= 2) {

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
                DataAndScore new_data(sample_ids[i]);
                new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].f);
                ret.train_data.pb(new_data);
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
                DataAndScore new_data(sample_ids[i]);
                new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].f);
                ret.test_data.pb(new_data);
            }
            */
        }
        sample_ids.clear();

        sample_ids = get_smallest_f(n);

        //test

        for(int i = 0;i<test_num_samples;i++)
        {
            DataAndScore new_data(sample_ids[i]);
            new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].f);
            ret.test_data.pb(new_data);
        }
    }




    ret.print();

    return ret;
}



/*
firstOrderDataset<DataAndScore> init_smallest_train_set(int n)
{
    firstOrderDataset<DataAndScore> ret;
    ret.n = n;
    //train
    vector<int> function_ids ;

    int train_num_samples = 256;
    int test_num_samples = (1<<(1<<n));

    vector<f_and_score> sample_ids = get_smallest_f(n, train_num_samples);

    //sample_ids.pb(0);
    //sample_ids.pb((1<<(1<<n))-1);

    //train
    for (int i = 0; i < train_num_samples; i++)
    {
        DataAndScore new_data(sample_ids[i].size, sample_ids[i].num_solutions);
        new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].f);
        ret.train_data.pb(new_data);
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
            sample_ids.pb(f_and_score(sample_id, _get_opt_decision_tree_score(n, sample_id)));
        }
    }
    else
    {
        assert(0);
    }

    //test

    for(int i = 0;i<test_num_samples;i++)
    {
        DataAndScore new_data(sample_ids[i].size, sample_ids[i].num_solutions);
        new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].f);
        ret.test_data.pb(new_data);
    }

    ret.print();

    return ret;
}
*/

void init_exaustive_table_with_unary_output(Data &f_data, int n, int f_id)
{
    f_data.init_exaustive_table_with_unary_output(n, f_id);
}

void init_exaustive_table_with_unary_output(DataAndScore &f_data, int n, int f_id)
{
    f_data.init_exaustive_table_with_unary_output(n, f_id);

    dp_decision_tree<Data> decision_tree_solver;
    net::parameters cutoff_parametes;
    DecisionTreeScore size_opt = decision_tree_solver.synthesize_decision_tree_and_get_size
            (cutoff_parametes, n, f_data, optimal);

    f_data.score = size_opt;
}



template<typename datatype>
firstOrderDataset<datatype> initFirstOrderDatasets(int n, int train_num_samples, int test_num_samples)
{
    firstOrderDataset<datatype> ret;

    vector<long long> sample_ids;

    //common
    if(n<=4) {

        for (int i = 0; i < (1 << (1 << n)); i++) {
            sample_ids.pb(i);
        }

        //shuffle(sample_ids.begin(), sample_ids.end(), std::default_random_engine(0));

    }
    else if(n<=6)
    {
        for(int i = 0; i < test_num_samples;i++)
        {
            long long first = (long long)rand(0, (1<<31)) << 31;
            long long second = rand(0, (1 << 31));

            long long sample_id = (first + second);

            //cout << first << " " << second << " "<< sample_id << endl;
            sample_ids.pb(sample_id);
        }
    }
    else
    {
        assert(0);
    }

    //train
    for (int i = 0; i < train_num_samples; i++) {

        datatype next_f;

        init_exaustive_table_with_unary_output(next_f, n, sample_ids[i]);

        ret.train_data.pb(next_f);
    }

    //test

    for(int i = 0;i<test_num_samples;i++)
    {
        datatype next_f;

        init_exaustive_table_with_unary_output(next_f, n, sample_ids[i]);

        ret.test_data.pb(next_f);
    }

    return ret;
}

template<typename datatype>
firstOrderDataset<datatype> completeFirstOrderTestDataset(int n)
{
    return initFirstOrderDatasets<datatype>(n, 0, (1<<(1<<n)));
}

vector<vector<Data> > initSecondOrderDataset(vector<Data> data, int subDataset_size)
{
    vector<vector<Data> > ret;
    for(int i = 0, meta_data_id = 0;i<data.size();meta_data_id++) {
        ret.pb(vector<Data>());
        for (int k = 0; k < subDataset_size && i < data.size(); i++, k++) {
            ret[meta_data_id].pb(data[i]);
        }
    }
    return ret;
}

secondOrderDataset<Data> initSecondOrderDatasets(firstOrderDataset<Data> data)
{
    secondOrderDataset<Data> ret;
    int subDataset_size = 2;
    ret.train_meta_data = initSecondOrderDataset(data.train_data, subDataset_size);
    ret.test_meta_data = initSecondOrderDataset(data.test_data, subDataset_size);

    return ret;
}

template<typename datatype>
int get_swap_count(vector<datatype> f_data, net the_net, int leaf_iter)
{

    Policy policy(f_data, false);
    firstOrderLearning<datatype> learning_algorighm = firstOrderLearning<datatype>();

    typename firstOrderLearning<datatype>::evaluate_learner_parameters params;
    params.iter_cutoff = leaf_iter;
    params.threshold = 0.01;

    net_and_score score =
            learning_algorighm.evaluate_learner
                    (the_net, f_data, false, params, &net::softPriorityTrain);

    return policy.get_swap_count(&score);

}

template<typename datatype>
class bitvectorFunctionSolver: public secondOrderLearning<datatype>
{
    typedef secondOrderLearning<datatype> base;
public:
    int root_iter = 60, leaf_iter = 60;

    bitvectorFunctionSolver(int _leaf_iter): secondOrderLearning<datatype>()
    {
        assert(_leaf_iter >= 1);
        leaf_iter = _leaf_iter;
    }

    firstOrderDataset<datatype> first_order_data;
    secondOrderDataset<datatype> meta_data;

    bool first_order_data_inited = false;
    bool meta_data_inited = false;

    void initFirstOrderData(int n, firstOrderDataset<datatype> _first_order_data)
    {
        first_order_data = _first_order_data;
        first_order_data_inited = true;
    }

    void initMetaData(int n, firstOrderDataset<datatype> _first_order_data)
    {


        first_order_data = _first_order_data;
        first_order_data_inited = true;

        meta_data = initSecondOrderDatasets(first_order_data);

        meta_data_inited = true;
    }

    const double treshold = 0.4;
    double min_treshold = treshold/3;

    net_and_score train_to_order_by_difficulty(int n, bool print)
    {
        assert(first_order_data_inited);

        net_and_score learner = net_and_score(net(n, 2*n, 1));

        order_tasks_by_difficulty_via_mutli_task_learning(learner, first_order_data, print, leaf_iter, treshold);
    }

    net_and_score train_to_order_neural_errors
            (int n, net_and_score learner, typename firstOrderLearning<datatype>::learning_parameters param, bool print)
    {
        assert(first_order_data_inited);

        base::order_neural_errors(learner, first_order_data.train_data, param, print);

        return learner;
    }

    net_and_score train_to_learn(int n, typename firstOrderLearning<datatype>::learning_parameters param, bool print)
    {
        assert(first_order_data_inited);

        srand(time(0));
        net_and_score learner = net_and_score(net(n, 2*n, 1));

        return train_to_learn(n, learner, param, print);
    }

    net_and_score train_to_learn(int n, net_and_score learner, typename firstOrderLearning<datatype>::learning_parameters param, bool print)
    {
        assert(first_order_data_inited);

        base::meta_learn(learner, first_order_data.train_data, param, print);

        min_treshold = learner.max_error;
        leaf_iter = learner.max_leaf_iter;

        assert(leaf_iter != 0);

        return learner;
    }


    void test_to_learn(net_and_score learner)
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
        meta_net_and_score learner = net_and_score(net(n, 2*n, 1));

        return train_to_meta_learn(n, learner);
    }

    meta_net_and_score train_to_meta_learn(int n, meta_net_and_score learner)
    {
        assert(meta_data_inited);

        typename firstOrderLearning<datatype>::evaluate_learner_parameters param;
        param.iter_cutoff = leaf_iter;
        param.threshold = treshold;

        base::learn_to_meta_learn(learner, meta_data.train_meta_data, root_iter, param);

        min_treshold = learner.max_error;
        leaf_iter = learner.max_leaf_iter;

        assert(leaf_iter != 0);

        return learner;
    }

    void test_to_meta_learn(meta_net_and_score learner)
    {
        assert(meta_data_inited);

        typename firstOrderLearning<datatype>::evaluate_learner_parameters params;
        params.iter_cutoff = leaf_iter;
        params.threshold = min_treshold;

        meta_net_and_score rez =
                base::evaluate_meta_learner(learner, meta_data.test_meta_data, true, root_iter, params);

        learner.printWeights();
        cout << "rez = " << rez.print() <<endl;
    }

    /*void print_test_data_ordering(int n, vector<net*> ensamble_nets, int num_iter)
    {

        assert(first_order_data_inited);

        vector<pair<DecisionTreeScore, int> > train_error_ordering;
        vector<pair<DecisionTreeScore, int> > opt_ordering;

        vector<DecisionTreeScore> opt_scores = vector<DecisionTreeScore>
                (first_order_data.test_data.size());

        for(int i = 0; i < first_order_data.test_data.size();i++)
        {
            //ensamble neural training error
            double local_error = 0;
            for (int j = 0; j < ensamble_nets.size(); j++) {
                assert(j < ensamble_nets.size());
                net leaf_learner = ensamble_nets[j];
                net::parameters leaf_parameters = cutoff_param(leaf_iter, 0.01);
                //cout << "init potential branch train" << endl;
                leaf_learner.train(&first_order_data.test_data[i], leaf_parameters, &net::softPriorityTrain);
                //cout << "end potential branch train" << endl;
                local_error += (leaf_learner.get_error_of_data(&first_order_data.test_data[i]).s);
            }

            //opt error
            dp_decision_tree<datatype> decision_tree_solver;
            net::parameters cutoff_parametes;
            DecisionTreeScore size_opt = decision_tree_solver.synthesize_decision_tree_and_get_size
                    (cutoff_parametes, n, first_order_data.test_data[i], optimal);

            train_error_ordering.pb(mp(DecisionTreeScore(local_error), i));
            opt_ordering.pb(mp(size_opt, i));
            opt_scores[i] = size_opt;
        }
        sort_v(opt_ordering);
        sort_v(train_error_ordering);
        for(int i = 0;i<opt_ordering.size();i++) {
            cout << first_order_data.test_data[train_error_ordering[i].s].printConcatinateOutput() << "\t";
            cout << fixed << setprecision(4) << (double) train_error_ordering[i].f.size << "\t | \t";

            cout << first_order_data.test_data[train_error_ordering[i].s].printConcatinateOutput() << "\t";
            cout << (int) opt_scores[train_error_ordering[i].s].size << "\t";
            cout << (int) opt_scores[train_error_ordering[i].s].num_solutions << "\t | \t";

            cout << endl;
        }

        //only one network test
        assert(ensamble_nets.size() == 1);



        int num_training_swaps = get_swap_count<datatype>(first_order_data.train_data, ensamble_nets[0], leaf_iter);
        int num_testing_swaps = get_swap_count<datatype>(first_order_data.test_data, ensamble_nets[0], leaf_iter);

        cout << "#testing_swaps = " << num_testing_swaps << endl;
        cout << endl;

        fout_experiment << "\t" << first_order_data.train_data.size() <<"\t" << num_testing_swaps << "\t" << num_training_swaps << endl;


    }*/

    /*meta_net_and_score learn_to_generalize(meta_net_and_score learner)
    {

    }*/
};


vector<vector<pair<pair<double, double>, DecisionTreeScore> > > get_ensamble_neural_error_buckets
        (int n, firstOrderDataset<DataAndScore> first_order_data, net::parameters params)
{
    vector<vector<pair<pair<double, double>, DecisionTreeScore> > > ret;

    vector<DataAndScore> sorted;

    vector<net*> ensamble_nets = params.progressive_ensamble_nets[n];

    int leaf_iter = params.get_iteration_count(n);

    sorted = first_order_data.train_data;

    sort_v(sorted);

    first_order_data.train_data = sorted;


    for(int i = 0; i < ensamble_nets.size(); i++)
    {
        vector<double> sums;
        vector<int> count;
        bool prev_score_defined = false;
        DecisionTreeScore prev_score;

        //cout << "size = " << first_order_data.train_data.size() <<endl;


        vector<pair<pair<double, double>, DecisionTreeScore> > overlapping;

        cout << "first_order_data.train_data.size() = " << first_order_data.train_data.size() <<endl;

        for(int j = 0; j < first_order_data.train_data.size();j++)
        {


            net leaf_learner = ensamble_nets[i];
            net::parameters leaf_parameters = cutoff_param(leaf_iter, 0.01);
            leaf_learner.train(&first_order_data.train_data[j], leaf_parameters, &net::softPriorityTrain);
            double error = leaf_learner.get_error_of_data(&first_order_data.train_data[j]).s;

            cout << first_order_data.train_data[j].printConcatinateOutput() << "\t" << first_order_data.train_data[j].score.size << "\t" << error << endl;

            if(prev_score_defined)
            {
                if(prev_score.size == first_order_data.train_data[j].score.size)
                {
                    int last_id = sums.size()-1;
                    overlapping[last_id].f.f = min(error, overlapping[last_id].f.f);
                    overlapping[last_id].f.s = max(error, overlapping[last_id].f.s);
                    sums[last_id] += error;
                    count[last_id] += 1;
                }
                else
                {
                    overlapping.pb(mp(mp(error, error), first_order_data.train_data[j].score));
                    sums.pb(error);
                    count.pb(1);
                    prev_score = first_order_data.train_data[j].score;
                }
            }
            else
            {
                overlapping.pb(mp(mp(error, error), first_order_data.train_data[j].score));
                sums.pb(error);
                count.pb(1);
                prev_score = first_order_data.train_data[j].score;
                prev_score_defined = true;
            }

        }

        assert(sums.size() == count.size());

        cout <<"n = " << n << " sums.size() == " << sums.size() <<endl;

        for(int local_i = 0;local_i<sums.size()-1;local_i++)
        {
            cout << "local_i = " << local_i <<endl;
            cout << "sums[local_i] = " << sums[local_i]<<" | count[local_i] = " << count[local_i] <<" | ";
            double at_now = sums[local_i]/count[local_i];
            double at_next = sums[local_i]/count[local_i];
            double mid = (at_now+at_next)/2;
            if(overlapping[local_i].f.s < overlapping[local_i+1].f.f)
            {
                //all good
            }
            else
            {
                overlapping[local_i].f.s = mid;
                overlapping[local_i + 1].f.f = mid;
            }
            cout << "at_now = " << at_now <<" | " << "at_next = " << at_next <<" | mid = " << mid << endl;
        }

        overlapping[sums.size()-1].f.s = 1;

        ret.pb(overlapping);

    }

    cout << "MODEL BRACKETS:" <<endl;

    for(int i = 0;i<ret.size();i++)
    {
        for(int j = 0;j<ret[i].size();j++)

        {
            cout << ret[i][j].f.f <<" "<< ret[i][j].f.s <<" "<< ret[i][j].s.size <<" | " <<endl;
        }
    }


    return ret;
}

vector<int> hidden_layer_block(int width, int depth)
{
    vector<int> ret;
    for(int i = 0;i<depth;i++)
    {
        ret.pb(width);
    }
    return ret;
}

vector<vector<vector<pair<pair<double, double>, DecisionTreeScore> > > > get_progressive_ensamble_neural_error_buckets
        (int n, vector<firstOrderDataset<DataAndScore> > first_order_datasets, net::parameters params)
{

    vector<vector<vector<pair<pair<double, double>, DecisionTreeScore> > > > ret;
    ret.pb(vector<vector<pair<pair<double, double>, DecisionTreeScore> > > ());
    for(int i = 1; i <= n; i++)
    {
        ret.pb(get_ensamble_neural_error_buckets
                       (i, first_order_datasets[i], params));
    }
    return ret;
}

template<typename datatype>
net_and_score improve_initialization(
        int local_n,
        net_and_score init_initialization,
        firstOrderDataset<datatype> first_order_dataset,
        typename firstOrderLearning<datatype>::learning_parameters param,
        bool print
)
{

    firstOrderLearning<datatype> trainer;

    net_and_score improved_initialization  =
            trainer.order_neural_errors
                    (init_initialization, first_order_dataset.train_data, param, false);

    net_and_score test_good_initialization =
            trainer.evaluate_learner
                    (improved_initialization, first_order_dataset.test_data, false, param, &net::softPriorityTrain);

    Policy test_policy = Policy(first_order_dataset.test_data);
    test_policy.update(&test_good_initialization);

    if(print)
    {
        trainer.print_ordering(test_policy, first_order_dataset.test_data, test_good_initialization.tarjans);

        test_good_initialization.printWeights();
    }

    return test_good_initialization;
}

template<typename datatype>
class generalizationDataset: public firstOrderDataset<firstOrderDataset<datatype> >
{

    /*

    vector<
        pair<
            vector<datatype> train_data,
            vector<datatype> test_data
            >
        > train_data;

    vector<
        pair<
            vector<datatype> train_data,
            vector<datatype> test_data
            >
        > test_data;

     */

};

vector<DataAndScore> select_all_functions(int n)
{
    assert(n<=3);

    vector<DataAndScore> ret;

    vector<f_and_score> sample_ids = get_smallest_f(n);

    if(n == 3)
    {
        srand(0);
        for(int i = 0;i<sample_ids.size();i++)
        {
            if(sample_ids[i].size < 7)
            {
                DataAndScore new_data(sample_ids[i]);
                new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].f);

                ret.pb(new_data);
            }
            else if(/*sample_ids[i].size <= 9 && */__builtin_popcount(i) <= 3)
            {
                DataAndScore new_data(sample_ids[i]);
                new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].f);

                ret.pb(new_data);
            }
        }
    }
    else
    {
        assert(0);
    }

    return ret;
}

firstOrderDataset<DataAndScore> split_into_first_order_dataset(vector<DataAndScore> all_functions, double training_set_size)
{
    firstOrderDataset<DataAndScore> ret;

    for(int i = 0;i<all_functions.size();i++)
    {
        if(rand(0, 100) < training_set_size*100)
        {
            ret.train_data.pb(all_functions[i]);
        }
        else
        {
            ret.test_data.pb(all_functions[i]);
        }
    }

    return ret;
}

vector<DataAndScore> take_percentage(vector<DataAndScore> all_data, double take_percent)
{
    vector<DataAndScore> ret;
    for(int i = 0;i<all_data.size();i++)
    {
        if(rand(0, 100) < take_percent*100)
        {
            ret.pb(all_data[i]);
        }
    }

    return ret;
}

vector<firstOrderDataset<DataAndScore> > sample_partitions(vector<DataAndScore> all_data, int num_samples)
{
    vector<firstOrderDataset<DataAndScore> > ret;

    for(int i = 0;i<num_samples;i++)
    {
        ret.pb(split_into_first_order_dataset(all_data, 0.5));
    }



    return ret;
}

void multi_task_learning_for_neural_error_ordering()
{
    int local_n = 3;

    vector<DataAndScore> all_functions = select_all_functions(local_n);

    srand(0);
    firstOrderDataset<DataAndScore> first_order_dataset = split_into_first_order_dataset(all_functions, 0.5);

    //firstOrderDataset<firstOrderDataset<datatype> >
    generalizationDataset<DataAndScore> holy_grail;

    int num_samples = 4;
    holy_grail.test_data = sample_partitions(first_order_dataset.test_data, num_samples);
    for (double percent_of_training_set = 0.3; percent_of_training_set <= 1; percent_of_training_set += 0.3)
    {

        vector<DataAndScore> sub_training_set = take_percentage(first_order_dataset.train_data,
                                                                percent_of_training_set);

        holy_grail.train_data = sample_partitions(sub_training_set, num_samples);

        for(int num_amphibian_iter = 10; num_amphibian_iter <= 30 ; num_amphibian_iter+=10)
        {

            for(int repeat = 0; repeat < 3; repeat++) {
                net_and_score at_init = net_and_score(net(local_n, 2 * local_n, 1));

                //param definition;
                typename firstOrderLearning<DataAndScore>::learning_parameters param;
                param.at_initiation_of_parameters_may_21st_10_52_pm();

                param.leaf_iter_init = 18;
                param.max_boundary_size = 5;
                param.max_num_iter = 10;


                int sum = 0;

                for (int i = 0; i < holy_grail.test_data.size(); i++) {
                    cout << "-----------------------------------------------------------------" << endl;
                    cout << "NOW PRE-TESTING ON FIRST_ORDER_DATASET.TEST[" << i << "]" << endl;
                    net_and_score result = improve_initialization<DataAndScore>(local_n, at_init,
                                                                                holy_grail.test_data[i], param, true);
                    sum += result.sum_ordering_error;
                }

                double pre_train_test_erorr = (double) sum / holy_grail.test_data.size();
                cout << "pre_train_test_error = " << pre_train_test_erorr << endl;


                for (int i = 0; i < num_amphibian_iter; i++) {
                    int first_order_dataset_id = rand(0, holy_grail.train_data.size() - 1);

                    cout << "-----------------------------------------------------------------" << endl;
                    cout << "NOW TRAINING ON FIRST_ORDER_DATASET.TRAIN[" << first_order_dataset_id << "]" << endl;

                    at_init = improve_initialization<DataAndScore>(local_n, at_init,
                                                                   holy_grail.train_data[first_order_dataset_id], param,
                                                                   false);
                }


                sum = 0;

                for (int i = 0; i < holy_grail.train_data.size(); i++) {
                    cout << "-----------------------------------------------------------------" << endl;
                    cout << "NOW TESTING ON FIRST_ORDER_DATASET.TRAIN[" << i << "]" << endl;
                    net_and_score result = improve_initialization<DataAndScore>(local_n, at_init,
                                                                                holy_grail.train_data[i], param, false);
                    sum += result.sum_ordering_error;
                }

                double post_train_train_error = (double) sum / holy_grail.train_data.size();
                cout << "post_train_train_error = " << post_train_train_error << endl;

                sum = 0;

                for (int i = 0; i < holy_grail.test_data.size(); i++) {
                    cout << "-----------------------------------------------------------------" << endl;
                    cout << "NOW TESTING ON FIRST_ORDER_DATASET.TEST[" << i << "]" << endl;
                    net_and_score result = improve_initialization<DataAndScore>(local_n, at_init,
                                                                                holy_grail.test_data[i], param, true);
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

template<typename datatype>
vector<net*> get_local_ensamble(
        int local_n, vector<int> hidden_layer_width, vector<int> hidden_layer_depth,
        int num_ensambles, vector<firstOrderDataset<datatype> > first_order_datasets, net ensamble_progressive_nets[10][10],
        vector<typename firstOrderLearning<datatype>::learning_parameters> all_params)
{
    vector<net*> local_ensamble;

    for(int ensamble_id = 0; ensamble_id < num_ensambles; ensamble_id++)
    {
        srand(time(0));

        vector<int> hidden_layer_widths = hidden_layer_block(hidden_layer_width[local_n], hidden_layer_depth[local_n-1]);

        net_and_score radnom_init = net_and_score(net(local_n, hidden_layer_widths, 1));

        net_and_score improved_initialization = improve_initialization(local_n, radnom_init, first_order_datasets[local_n], all_params[ensamble_id], true);

        ensamble_progressive_nets[local_n][ensamble_id] = improved_initialization;

        local_ensamble.pb(&ensamble_progressive_nets[local_n][ensamble_id]);
    }

    return local_ensamble;
}

class latentDecisionTreeExtractor: public secondOrderLearning<Data>
{
public:

    latentDecisionTreeExtractor(): secondOrderLearning()
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

    class SynthesizersScoreSummary: public net
    {
    public:
        SynthesizerScoreSum base_synthesizer_score;
        SynthesizerScoreSum other_synthesizer_score;
        SynhtesizerScoreCoparitor comparitor;

        SynthesizersScoreSummary(){}

        SynthesizersScoreSummary(net self): net(self)
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

    /*
    class SynthesizersScoreSummary: public net
    {
    public:
        double max_ratio = 1;
        int max_delta = 0;
        int sum_sizes = 0;
        int sum_errors = 0;
        int num_non_opt = 0;
        int num_neural_guided_non_opt = 0;

        vector<Data> non_opt;

        int sum_entropy_errors = 0;
        int num_entropy_non_opt = 0;

        SynthesizersScoreSummary()
        {

        }

        SynthesizersScoreSummary(net self): net(self)
        {

        }

        bool operator < (const SynthesizersScoreSummary& ret) const
        {
            return sum_errors < ret.sum_errors;
        }

        string print_sums()
        {
            return ""
        }

        string print()
        {
            return "sum_errors = \t" + std::to_string(sum_errors)+
                   ", num_non_opt = " + std::to_string(num_non_opt);
        }

        string print_entropy()
        {
            return "sum_entropy_errors = \t" + std::to_string(sum_entropy_errors)+
                   ", num_entropy_non_opt = " + std::to_string(num_entropy_non_opt);
        }

        void clear_score()
        {
            max_ratio = 1;
            max_delta = 0;
            sum_sizes = 0;
            sum_errors = 0;
            num_non_opt = 0;

            non_opt.clear();

            sum_entropy_errors = 0;
            num_entropy_non_opt = 0;
        }
    };
     */

    SynthesizersScoreSummary run_decision_tree_synthesis_comparison(
            SynthesizersScoreSummary meta_meta_net,
            bitvectorFunctionSolver<Data> bitvector_data,
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

                net::parameters param = cutoff_param(8, 0.01);
                int local_iter = leaf_net.train(&data[i][j], param, &net::softPriorityTrain);

                double local_error = leaf_net.get_error_of_data(&data[i][j]);

                if(print)
                {
                    cout << "f: " << data[i][j].printConcatinateOutput() << endl;
                    cout << "iters: " << local_iter <<endl;
                    cout << "error: " << local_error << endl;
                }*/


                SynthesizersScoreSummary new_leaf_net = meta_net;

                //tailor is the neural network that tailors towards inducing a special structure
                net::parameters tailor_param = cutoff_param(7, 0.01);
                tailor_param.track_dimension_model = true;
                tailor_param.neural_net = &new_leaf_net;

                dp_decision_tree<Data> decision_tree_solver;
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

                        ret.non_opt.pb(data[i][j]);
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
            bitvectorFunctionSolver<Data> bitvector_data,
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

            SynthesizersScoreSummary next = net_and_score(step(at, 0.35*temperature));

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
        cout << "IN latentDecisionTreeExtractor" <<endl <<endl;

        int leaf_iter = 30;

        bitvectorFunctionSolver<Data> bitvector_data(leaf_iter);

        //meta_net_and_score meta_meta_net = bitvector_data.train(n);
        SynthesizersScoreSummary meta_meta_net = net_and_score(net(n, 2*n, 1));

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
                bitvector_data.meta_data.train_meta_data.pb(train_meta_data[i]);
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

        cout << "In main train(..)" <<endl;
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
            vector<net*> progressive_nets,
            bitvectorFunctionSolver bitvector_data,
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
                net::parameters tailor_param = meta_cutoff(bitvector_data.root_iter, bitvector_data.leaf_iter);
                tailor_param.progressive_nets = progressive_nets;

                dp_decision_tree decision_tree_solver;
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

                    ret.non_opt.pb(data[i][j]);
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



    template<typename datatype>
    void train_library(
            int min_trainable_n,
            int max_n, vector<int> leaf_iters, vector<int> hidden_layer_width, vector<int> hidden_layer_depth,
            int num_ensambles, vector<firstOrderDataset<datatype> > first_order_datasets)
    {
        assert(leaf_iters.size() > max_n-1);
        assert(hidden_layer_width.size() > max_n-1);
        net ensamble_progressive_nets[10][10];
        vector<vector<net*> > ensamble_progressive_net_pointers;
        ensamble_progressive_net_pointers.pb(vector<net*>());// for n = 0;

        for(int local_n = 2; local_n < min_trainable_n; local_n++)
        {
            vector<net*> local_ensamble;
            for(int ensamble_id = 0; ensamble_id < num_ensambles; ensamble_id++) {
                srand(time(0));

                vector<int> hidden_layer_widths = hidden_layer_block(hidden_layer_width[local_n - 1], hidden_layer_depth[local_n-1]);

                SynthesizersScoreSummary meta_meta_net =
                        net_and_score(net(local_n - 1, hidden_layer_widths, 1));

                ensamble_progressive_nets[local_n - 1][ensamble_id] = meta_meta_net;
                local_ensamble.pb(&ensamble_progressive_nets[local_n - 1][ensamble_id]);
            }
            ensamble_progressive_net_pointers.pb(local_ensamble);
        }

        for(int local_n = min_trainable_n; local_n <= max_n; local_n++)
        {


            bool do_learn = (local_n == 4);

            vector<net*> local_ensamble =
                    get_local_ensamble(local_n-1, hidden_layer_width, hidden_layer_depth,
                                       num_ensambles, first_order_datasets,  do_learn, ensamble_progressive_nets);


            // (local_n, leaf_iters, hidden_layer_width, hidden_layer_depth,
            //       num_ensambles, first_order_datasets, do_learn, ensamble_progressive_net_pointers);

            ensamble_progressive_net_pointers.pb(local_ensamble);

            net::parameters cutoff_parameter = net::parameters(leaf_iters);
            cutoff_parameter.progressive_ensamble_nets = ensamble_progressive_net_pointers;

            if(do_learn)
            {
                cutoff_parameter.progressive_ensamble_neural_error_buckets =
                        get_progressive_ensamble_neural_error_buckets
                                (local_n - 1, first_order_datasets, cutoff_parameter);
            }

            assert(0);//fix this "if(local_n <= 3)"

            if(local_n <= 3)
            {
                continue;
            }


            bitvectorFunctionSolver<datatype> complete_bitvector_data(leaf_iters[local_n]);

            complete_bitvector_data.initFirstOrderData(local_n, first_order_datasets[local_n]);


            dp_decision_tree<datatype> decision_tree_solver;

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

void see_delta_w()
{
    const int n = 3;

    const int middle_layer = n;

    net walker = net(n, middle_layer, 1);

    vector<pair<net::PriorityTrainReport, int> > to_sort;

    vector<net> net_data;

    Data meta_task;

    net walker_init = net(n, middle_layer, 1);

    for(int i = 0; i<(1<<(1<<n))/8;i++)
    {
        Data the_data;
        the_data.init_exaustive_table_with_unary_output(n, i);

        net::parameters param = net::parameters(1, 1);//rate, batch_width

        walker.set_special_weights();
        //walker = walker_init;
        walker.save_weights();
        printOnlyBatch = false;
        cout << bitset<(1<<n)>(i).to_string()<<endl;
        param.track_dimension_model = false;
        param.accuracy = 0.25/2;
        to_sort.push_back(mp(walker.train(&the_data, param, &net::softPriorityTrain), i));

        net::test_score score = walker.test(&the_data, 0.25/2);

        assert(score.get_wrong() == 0);
        //walker.compare_to_save();
        //walker.printWeights();
        meta_task.push_back(to_bit_signature_vector((1<<n), i), walker.to_bit_signature_vector());
    }

    Data noramlized_meta_task;
    meta_task.normalize(noramlized_meta_task);

    /*for(int i = 0;i<meta_task.size();i++)
    {
        meta_task.printTest(i);
    }*/

    net net_learning_nets = net(noramlized_meta_task.numInputs, 2*noramlized_meta_task.numInputs, noramlized_meta_task.numOutputs);

    net::parameters param = net::parameters(1, 3);//rate, batch_width
    printItteration = true;
    param.track_dimension_model = false;
    param.accuracy = 0.0125*2;
    net_learning_nets.train(&noramlized_meta_task, param, &net::softPriorityTrain);

    Data resulting_data;

    int total_error = 0;
    for(int i = 0; i<noramlized_meta_task.size();i++)
    {
        vector<bit_signature> network_output = net_learning_nets.forwardPropagate(to_bit_signature_vector((1<<n), i), false);

        noramlized_meta_task.unnormalize(network_output);

        net output_network = net(n, middle_layer, 1, network_output);
        //output_network.printWeights();

        Data the_data;
        the_data.init_exaustive_table_with_unary_output(n, i);

        net::test_score score = output_network.test(&the_data, 0.5);
        for(int i = 0;i<score.correct_examples.size();i++)
        {
            cout << score.correct_examples[i];
        }
        total_error += score.get_wrong();
        cout << " #wrong = " << score.get_wrong() <<endl;
    }
    cout << total_error << " over " << noramlized_meta_task.size() << endl;

    /*
    for(int i = 0; i<(1<<(1<<n));i++)
    {
        Data the_data;
        the_data.init_exaustive_table_with_unary_output(n, i);

        net::parameters param = net::parameters(1, 1);//rate, batch_width

        walker.set_special_weights();
        //walker.save_weights();
        printOnlyBatch = true;
        cout << bitset<(1<<n)>(i).to_string()<<" ";
        to_sort.push_back(mp(walker.train(&the_data, param, &net::softPriorityTrain), i));
        //walker.compare_to_save();
    }
    cout << endl;
    sort_v(to_sort);
    for(int i = 0;i<to_sort.size();i++)
    {
        cout << bitset<(1<<n)>(to_sort[i].s).to_string() <<endl;
    }*/
}

class graph_plotter
{
public:
    void init()
    {
        printCycle = false;
        printItteration = false;
        printCleanItteration = false;

        print_delta_knowledge_graph = false;
        print_important_neurons = false;
        print_classify_neurons = false;
        print_implications = false;
        print_the_imporant_bit = false;

        print_discrete_model = false;
        printNewChildren = false;

        print_tree_synthesys = false;
        print_try_and_finally = false;
        printOnlyBatch = false;


        if(true)
        {

            if(false)
            {
                cout << "See delta_w" << endl;
                see_delta_w();
                return;
            }
            if(true)
            {
                multi_task_learning_for_neural_error_ordering();
            }
            if(false)
            {
                for(int num_iter = 20; ; num_iter += 10)
                    for(double training_set_size = 0.5; training_set_size <= 1 ; training_set_size+=0.1)
                    {
                        vector<int> leaf_iters;
                        leaf_iters.pb(-1);//no leaf_iter for n = 0;
                        leaf_iters.pb(8);// for n = 1;
                        leaf_iters.pb(16);// for n = 2;
                        leaf_iters.pb(18);// for n = 3;
                        leaf_iters.pb(32);// for n = 4;
                        leaf_iters.pb(64);// for n = 5;

                        vector<int> hidden_layer_width;
                        hidden_layer_width.pb(-1);//no leaf_iter for n = 0;
                        hidden_layer_width.pb(2);// for n = 1;
                        hidden_layer_width.pb(4);// for n = 2;
                        hidden_layer_width.pb(6);// for n = 3;
                        hidden_layer_width.pb(8);// for n = 4;
                        hidden_layer_width.pb(10);// for n = 5;


                        vector<int> hidden_layer_depth;
                        hidden_layer_depth.pb(-1);//no leaf_iter for n = 0;
                        hidden_layer_depth.pb(1);// for n = 1;
                        hidden_layer_depth.pb(1);// for n = 2;
                        hidden_layer_depth.pb(1);// for n = 3;
                        hidden_layer_depth.pb(1);// for n = 4;
                        hidden_layer_depth.pb(1);// for n = 5;

                        int min_trainable_n = 4;
                        int max_trainable_n = 4;

                        vector<firstOrderDataset<DataAndScore> > datasets;
                        for(int i = 0;i<min_trainable_n-1;i++)
                        {
                            datasets.pb(firstOrderDataset<DataAndScore>());//n < min_trainable_n
                        }

                        for(int i = min_trainable_n-1; i<max_trainable_n; i++)
                        {
                            datasets.pb(init_custom_data_and_difficulty_set(i, training_set_size*100, 60));
                        }


                        // need to remove assert (n<=3)
                        // datasets.pb(init_custom_data_and_difficulty_set(max_trainable_n));


                        //datasets.pb(init_random_test_set(max_trainable_n));

                        for(int repeat = 0; repeat < 5; repeat++) {

                            assert(min_trainable_n == max_trainable_n);
                            net ensamble_progressive_nets[10][10];

                            vector<typename firstOrderLearning<DataAndScore>::learning_parameters> all_params;
                            all_params.pb(typename firstOrderLearning<DataAndScore>::learning_parameters());
                            all_params[0].at_initiation_of_parameters_may_21st_10_52_pm();

                            all_params[0].leaf_iter_init = leaf_iters[3];
                            all_params[0].treshold = 0.4;
                            all_params[0].max_boundary_size = 5;
                            all_params[0].max_num_iter = num_iter;

                            /*int leaf_iter_init;

                            double treshold;*/

                            fout_experiment << num_iter <<"\t";

                            get_local_ensamble<DataAndScore>
                                    (min_trainable_n-1, hidden_layer_width, hidden_layer_depth,
                                     1, datasets, ensamble_progressive_nets, all_params);



                            latentDecisionTreeExtractor SpicyAmphibian = latentDecisionTreeExtractor();
                            /*SpicyAmphibian.train_library<DataAndScore>
                                    (min_trainable_n, max_trainable_n, leaf_iters, hidden_layer_width, hidden_layer_depth,
                                     ensamble_size, datasets);*/
                        }
                    }
                return;
            }
            if(false)
            {
                dp_decision_tree<Data> dper;
                printItteration = false;
                print_close_local_data_model = false;
                dper.run();
                return;
            }

            if(false)
            {
                string type = "longestSubstring_ai_is_1";
                Data the_data = Data();
                int n = 6, outputSize;
                the_data.generateData(n, outputSize, type);

                printItteration = true;
                net::parameters param = net::parameters(1.5, 3);
                param.num_stale_iterations = 10000;//7;
                param.set_iteration_count(16);//(16);
                param.ensamble_size = 1;//8;
                param.priority_train_f = &net::softPriorityTrain;

                decision_tree tree = decision_tree(&the_data, param);

                return ;
            }

            if(false) //via circuit
            {
                net::parameters param = net::parameters(1.5, 3);

                archaic_neural_designed_circuit two_nets;
                param.num_stale_iterations = 7;
                param.set_iteration_count(16);

                string type = "longestSubstring_ai_is_1";
                Data the_data = Data();
                int n = 4, outputSize;
                the_data.generateData(n, outputSize, type);

                two_nets.build_circuit_per_output_dimension(the_data, param);
                //two_nets.build_circuit_based_on_singletons(6, param, &net::softPriorityTrain);
                return;
            }


            if(false)
            {
                //single init root/local build
                archaic_neural_decision_tree tree = archaic_neural_decision_tree();
                //tree.local_single_build(10, net::parameters(1, 2));

                //return ;

                net::parameters param = net::parameters(3.79, 3);
                //net::parameters param = net::parameters(1, 3);

                param.num_stale_iterations = 3;
                int to_print = tree.init_root(8, param, &net::softPriorityTrain);

                cout << to_print << endl;

                int num = tree.print_gate(0);

                cout << "num gates = " << num <<endl;

                cout << endl;
                return ;
            }



            print_discrete_model = false;
            printItteration = false;
            printNewChildren = false;

            //for(double id = 0.5;id<=10;id+=0.5)
            for(int id_iter_count = 16; id_iter_count <= 16; id_iter_count*=2)
            {
                cout <<endl;
                cout << id_iter_count <<" ::: " << endl;
                for(int id_ensamble_size = 1;id_ensamble_size<=12;id_ensamble_size++)
                {
                    cout << id_ensamble_size << "\t" <<" :: " << "\t";
                    for(int i = 0;i<6;i++)
                    {
                        net::parameters param = net::parameters(1.5, 3);
                        param.num_stale_iterations = 7;
                        param.set_iteration_count(id_iter_count);
                        param.ensamble_size = id_ensamble_size;


                        string type = "longestSubstring_ai_is_1";
                        Data the_data = Data();
                        int n = 6, outputSize;
                        the_data.generateData(n, outputSize, type);

                        decision_tree tree = decision_tree(&the_data, param);
                        cout << tree.size <<"\t";

                        //archaic_neural_decision_tree tree = archaic_neural_decision_tree();
                        //cout << tree.init_root(6, param, &net::softPriorityTrain) <<"\t";

                        //int num = tree.print_gate(0);

                        //cout << "num gates = " << num <<endl;
                        //cout << endl <<endl;

                    }
                    cout << endl;
                }
            }


            //rate plotter
            /*double greates_rate = 3;
             for(double id = greates_rate;id>=1;id-=0.2)
             {
             cout << id << "\t" <<" :: " << "\t";
             for(int i = 0;i<40;i++)
             {
             cout << tree.local_single_build(5, id, greates_rate, 4) <<"\t";
             }
             cout << endl;
             }*/

            return ;
        }
        else
        {
            /*print_the_imporant_bit = false;
             for(double learning_rate = 3; true; learning_rate*=0.8)
             {
             cout << learning_rate << "\t" << ":" << "\t";
             for(int num_trials = 0; num_trials < 7; num_trials++)
             {
             archaic_neural_decision_tree tree = archaic_neural_decision_tree();
             int num_nodes = tree.init_root(8, learning_rate, 1, &net::queueTrain);
             cout  << num_nodes << "\t";
             }
             cout << endl;
             }*/
        }
    }
};

int main()
{

    print_all_dt_strings(3);
    return 0;

    //srand(time(0));
    clock_t count = clock();
    graph_plotter worker;
    worker.init();
    cout << "time elapsed = " << (double)(clock()-count)/CLOCKS_PER_SEC<<endl;
    return 0;
}



/*
 interesting thigs to try::

 fragment code into files
 stoping learning once a good dimension to cut is determiend
 do time and iteration tests based on learning rate, learning iteration termination, queue vs priority, different topologiez
 learning in batches of different size
 learning in weighted batcehs
 learning with gradients of learning rates


 John conway's learning
 Create iterativelly small machines. Have an algorithm that adjusts network topology.
 Select them based on some heuristic to do with learning them. For eg. which have highest gradients of decision tree size based on learning.



 */


/*
11111111	0.0677	 | 	11111111	1	 |
00110011	0.0894	 | 	00110011	3	 |
01010101	0.0899	 | 	01010101	3	 |
00001111	0.0934	 | 	00001111	3	 |
11001100	0.0941	 | 	11001100	3	 |
11110000	0.0972	 | 	11110000	3	 |
10101010	0.0986	 | 	10101010	3	 |
11110011	0.1305	 | 	11110011	5	 |
01110111	0.1311	 | 	01110111	5	 |
10111011	0.1332	 | 	10111011	5	 |
11011101	0.1348	 | 	11011101	5	 |
00000101	0.1447	 | 	00000101	5	 |
01011111	0.1457	 | 	01011111	5	 |
00111111	0.1459	 | 	00111111	5	 |
11101110	0.1474	 | 	11101110	5	 |
10001000	0.1491	 | 	10001000	5	 |
10100000	0.1498	 | 	10100000	5	 |
00000011	0.1498	 | 	00000011	5	 |
11000000	0.1506	 | 	11000000	5	 |
11111100	0.1506	 | 	11111100	5	 |
11101000	0.1506	 | 	11101000	11	 |
00001100	0.1515	 | 	00001100	5	 |
00001010	0.1532	 | 	00001010	5	 |
00100010	0.1543	 | 	00100010	5	 |
11111010	0.1545	 | 	11111010	5	 |
01000100	0.1562	 | 	01000100	5	 |
00110000	0.1578	 | 	00110000	5	 |
01110001	0.1608	 | 	01110001	11	 |
00101011	0.1617	 | 	00101011	11	 |
01010000	0.1658	 | 	01010000	5	 |
11010100	0.1669	 | 	11010100	11	 |
10110010	0.1676	 | 	10110010	11	 |
01001101	0.1699	 | 	01001101	11	 |
10001110	0.1728	 | 	10001110	11	 |
00010111	0.1746	 | 	00010111	11	 |
11001000	0.2633	 | 	11001000	7	 |
00001000	0.2658	 | 	00001000	7	 |
01000000	0.2742	 | 	01000000	7	 |
10101011	0.2758	 | 	10101011	7	 |
11111110	0.2773	 | 	11111110	7	 |
11111110	0.2773	 | 	11111110	7	 |
00100000	0.2798	 | 	00100000	7	 |
00001110	0.2847	 | 	00001110	7	 |
11011100	0.2879	 | 	11011100	7	 |
00110010	0.2956	 | 	00110010	7	 |
01010111	0.2997	 | 	01010111	7	 |
10111010	0.3002	 | 	10111010	7	 |
01010100	0.3035	 | 	01010100	7	 |
11110111	0.3056	 | 	11110111	7	 |
01111111	0.3078	 | 	01111111	7	 |
00000100	0.3110	 | 	00000100	7	 |
11111000	0.3136	 | 	11111000	7	 |
11111011	0.3163	 | 	11111011	7	 |
11101100	0.3183	 | 	11101100	7	 |
00000111	0.3213	 | 	00000111	7	 |
10001111	0.3224	 | 	10001111	7	 |
01001100	0.3260	 | 	01001100	7	 |
01001111	0.3302	 | 	01001111	7	 |
00010101	0.3313	 | 	00010101	7	 |
01110101	0.3319	 | 	01110101	7	 |
11010101	0.3320	 | 	11010101	7	 |
11011111	0.3323	 | 	11011111	7	 |
11001110	0.3386	 | 	11001110	7	 |
00010000	0.3470	 | 	00010000	7	 |
01110000	0.3495	 | 	01110000	7	 |
00001011	0.3495	 | 	00001011	7	 |
10001010	0.3501	 | 	10001010	7	 |
00111011	0.3504	 | 	00111011	7	 |
00000010	0.3509	 | 	00000010	7	 |
00101111	0.3582	 | 	00101111	7	 |
10110011	0.3632	 | 	10110011	7	 |
10101000	0.3701	 | 	10101000	7	 |
11110010	0.3709	 | 	11110010	7	 |
00010011	0.3718	 | 	00010011	7	 |
00110111	0.3727	 | 	00110111	7	 |
11001101	0.3745	 | 	11001101	7	 |
01010001	0.3748	 | 	01010001	7	 |
10001100	0.3750	 | 	10001100	7	 |
10100010	0.3768	 | 	10100010	7	 |
10111111	0.3831	 | 	10111111	7	 |
01000101	0.3862	 | 	01000101	7	 |
11010000	0.3976	 | 	11010000	7	 |
00100011	0.4062	 | 	00100011	7	 |
00110001	0.4093	 | 	00110001	7	 |
11110100	0.4110	 | 	11110100	7	 |
11000100	0.4138	 | 	11000100	7	 |
10101110	0.4186	 | 	10101110	7	 |
00001101	0.4228	 | 	00001101	7	 |
11100000	0.4428	 | 	11100000	7	 |
11111101	0.4634	 | 	11111101	7	 |
01110011	0.4897	 | 	01110011	7	 |
11101111	0.4920	 | 	11101111	7	 |
01011101	0.5359	 | 	01011101	7	 |
10000001	0.5969	 | 	10000001	11	 |
00001001	0.6024	 | 	00001001	9	 |
10100110	0.6128	 | 	10100110	11	 |
01101010	0.6334	 | 	01101010	11	 |
10011000	0.6411	 | 	10011000	9	 |
10111110	0.6464	 | 	10111110	9	 |
00101101	0.6468	 | 	00101101	11	 |
10111001	0.6481	 | 	10111001	9	 |
01110010	0.6728	 | 	01110010	7	 |
00111000	0.6802	 | 	00111000	9	 |
10100011	0.6812	 | 	10100011	7	 |
11101011	0.6817	 | 	11101011	9	 |
11011110	0.6915	 | 	11011110	9	 |
10100100	0.6932	 | 	10100100	9	 |
10001101	0.6961	 | 	10001101	7	 |
10101100	0.7017	 | 	10101100	7	 |
00100001	0.7095	 | 	00100001	9	 |
01111010	0.7105	 | 	01111010	9	 |
11100100	0.7110	 | 	11100100	7	 |
11100011	0.7117	 | 	11100011	9	 |
11011011	0.7141	 | 	11011011	11	 |
01101100	0.7143	 | 	01101100	11	 |
01100110	0.7143	 | 	01100110	7	 |
01100100	0.7143	 | 	01100100	9	 |
11110110	0.7163	 | 	11110110	9	 |
01111000	0.7165	 | 	01111000	11	 |
11100110	0.7170	 | 	11100110	9	 |
01100001	0.7215	 | 	01100001	13	 |
00011000	0.7271	 | 	00011000	11	 |
01001010	0.7274	 | 	01001010	9	 |
01000010	0.7274	 | 	01000010	11	 |
00111100	0.7329	 | 	00111100	7	 |
00011110	0.7356	 | 	00011110	11	 |
00101110	0.7371	 | 	00101110	7	 |
10101101	0.7375	 | 	10101101	9	 |
00111010	0.7398	 | 	00111010	7	 |
01101011	0.7402	 | 	01101011	13	 |
00011010	0.7446	 | 	00011010	9	 |
10101001	0.7447	 | 	10101001	11	 |
10010100	0.7470	 | 	10010100	13	 |
01011000	0.7478	 | 	01011000	9	 |
11000001	0.7502	 | 	11000001	9	 |
00101100	0.7511	 | 	00101100	9	 |
10010111	0.7516	 | 	10010111	13	 |
10010011	0.7516	 | 	10010011	11	 |
11100001	0.7535	 | 	11100001	11	 |
11100101	0.7535	 | 	11100101	9	 |
11100111	0.7593	 | 	11100111	11	 |
10010010	0.7606	 | 	10010010	13	 |
10011010	0.7606	 | 	10011010	11	 |
10110001	0.7615	 | 	10110001	7	 |
01000001	0.7638	 | 	01000001	9	 |
11010111	0.7715	 | 	11010111	9	 |
01011100	0.7715	 | 	01011100	7	 |
00011001	0.7727	 | 	00011001	9	 |
01100011	0.7730	 | 	01100011	11	 |
10110101	0.7738	 | 	10110101	9	 |
00110100	0.7746	 | 	00110100	9	 |
00100100	0.7746	 | 	00100100	11	 |
11101001	0.7777	 | 	11101001	13	 |
10111000	0.7791	 | 	10111000	7	 |
00101001	0.7792	 | 	00101001	13	 |
00011011	0.7794	 | 	00011011	7	 |
01011011	0.7801	 | 	01011011	9	 |
00111001	0.7805	 | 	00111001	11	 |
00100110	0.7807	 | 	00100110	9	 |
01101111	0.7815	 | 	01101111	9	 |
01110100	0.7847	 | 	01110100	7	 |
10010101	0.7877	 | 	10010101	11	 |
00100101	0.7893	 | 	00100101	9	 |
10000100	0.7911	 | 	10000100	9	 |
10011101	0.7912	 | 	10011101	9	 |
10010000	0.7926	 | 	10010000	9	 |
01101000	0.7956	 | 	01101000	13	 |
10100111	0.7970	 | 	10100111	9	 |
01001011	0.8007	 | 	01001011	11	 |
11011010	0.8075	 | 	11011010	9	 |
10000010	0.8092	 | 	10000010	9	 |
10011110	0.8099	 | 	10011110	13	 |
00100111	0.8109	 | 	00100111	7	 |
01101001	0.8134	 | 	01101001	15	 |
11011001	0.8170	 | 	11011001	9	 |
10001001	0.8177	 | 	10001001	9	 |
01001110	0.8182	 | 	01001110	7	 |
10010001	0.8215	 | 	10010001	9	 |
10011001	0.8224	 | 	10011001	7	 |
11111001	0.8240	 | 	11111001	9	 |
10111101	0.8271	 | 	10111101	11	 |
00011101	0.8329	 | 	00011101	7	 |
00111110	0.8425	 | 	00111110	9	 |
11100010	0.8428	 | 	11100010	7	 |
11011000	0.8449	 | 	11011000	7	 |
10000111	0.8465	 | 	10000111	11	 |
10000011	0.8465	 | 	10000011	9	 |
01100010	0.8467	 | 	01100010	9	 |
11001001	0.8467	 | 	11001001	11	 |
11001011	0.8509	 | 	11001011	9	 |
10111100	0.8541	 | 	10111100	9	 |
00110101	0.8543	 | 	00110101	7	 |
01010011	0.8564	 | 	01010011	7	 |
00101000	0.8566	 | 	00101000	9	 |
01111100	0.8595	 | 	01111100	9	 |
10110111	0.8622	 | 	10110111	9	 |
01111110	0.8658	 | 	01111110	11	 |
01010010	0.8674	 | 	01010010	9	 |
01001001	0.8679	 | 	01001001	13	 |
11010010	0.8687	 | 	11010010	11	 |
01100111	0.8737	 | 	01100111	9	 |
01000110	0.8775	 | 	01000110	9	 |
11000110	0.8775	 | 	11000110	11	 |
10000110	0.8775	 | 	10000110	13	 |
11000011	0.8793	 | 	11000011	7	 |
01000011	0.8816	 | 	01000011	9	 |
01111001	0.8837	 | 	01111001	13	 |
00010110	0.8846	 | 	00010110	13	 |
10010110	0.8846	 | 	10010110	15	 |
01010110	0.8846	 | 	01010110	11	 |
11010110	0.8846	 | 	11010110	13	 |
10100101	0.8922	 | 	10100101	7	 |
10100001	0.8922	 | 	10100001	9	 |
10011111	0.8947	 | 	10011111	9	 |
01111101	0.8955	 | 	01111101	9	 |
11010001	0.8966	 | 	11010001	7	 |
01101101	0.8969	 | 	01101101	13	 |
01100101	0.8976	 | 	01100101	11	 |
01100000	0.8982	 | 	01100000	9	 |
11001010	0.8994	 | 	11001010	7	 |
01111011	0.9038	 | 	01111011	9	 |
00111101	0.9039	 | 	00111101	9	 |
00110110	0.9107	 | 	00110110	11	 |
10011011	0.9126	 | 	10011011	9	 |

Network =
{
    {
        { {0.343846, 0.878073, -0.599613}, {-0.287050} },
        { {-0.038455, -0.055753, -0.045319}, {0.308151} },
        { {0.121851, -0.408460, -0.550222}, {-0.011402} },
        { {0.018635, 0.018354, 0.006582}, {0.162808} },
        { {-0.097966, -0.053717, 0.064828}, {0.048662} },
        { {-0.459572, -0.317840, -0.487781}, {0.378314} }
    },
    {
        { {0.584033, -6.756721, 0.167906, 3.677181, 3.568711, -0.948333}, {0.419629} }
    }
}
rez = max_error = 	0.912614	 max_iter = 15
ensamble_size = 1
hidden_layer_width:
n_in = 2	hidden_width 4
n_in = 3	hidden_width 6
leaf_iters:
n_in = 2	leaf_iter 8
n_in = 3	leaf_iter 16
opt_to_neural:: base = {	sum = 	918	}; compare_with = {	sum = 	992	}; comparitor = {	sum_dist = 	74	, max_dist = 	6	| avg_ratio = 	1.081688	, max_ratio = 	1.461538	}
neural_to_entropy:: base = {	sum = 	992	}; compare_with = {	sum = 	1074	}; comparitor = {	sum_dist = 	82	, max_dist = 	12	| avg_ratio = 	1.110892	, max_ratio = 	2.333333	}

 */


/**
SHOULD BE COOL


 BEFORE TRAINING:

11111111	0.0947	 | 	11111111	1	1	 |
00000000	0.1027	 | 	00000000	1	1	 |
01010101	0.4857	 | 	01010101	3	1	 |
11110000	0.5126	 | 	11110000	3	1	 |
10101010	0.5160	 | 	10101010	3	1	 |
01010000	0.5219	 | 	01010000	5	2	 |
00001111	0.5257	 | 	00001111	3	1	 |
00000101	0.5259	 | 	00000101	5	2	 |
10100000	0.5264	 | 	10100000	5	2	 |
01011111	0.5274	 | 	01011111	5	2	 |
11001100	0.5281	 | 	11001100	3	1	 |
11110101	0.5293	 | 	11110101	5	2	 |
01000100	0.5297	 | 	01000100	5	2	 |
11011101	0.5299	 | 	11011101	5	2	 |
00110000	0.5304	 | 	00110000	5	2	 |
00010000	0.5306	 | 	00010000	7	6	 |
00001100	0.5313	 | 	00001100	5	2	 |
00000100	0.5329	 | 	00000100	7	6	 |
11111010	0.5334	 | 	11111010	5	2	 |
11001111	0.5338	 | 	11001111	5	2	 |
01001101	0.5341	 | 	01001101	11	12	 |
11111100	0.5353	 | 	11111100	5	2	 |
11101110	0.5356	 | 	11101110	5	2	 |
11011111	0.5359	 | 	11011111	7	6	 |
11111101	0.5365	 | 	11111101	7	6	 |
00001010	0.5366	 | 	00001010	5	2	 |
11111110	0.5369	 | 	11111110	7	6	 |
00100000	0.5371	 | 	00100000	7	6	 |
10001000	0.5373	 | 	10001000	5	2	 |
10000000	0.5374	 | 	10000000	7	6	 |
01000000	0.5377	 | 	01000000	7	6	 |
00010001	0.5378	 | 	00010001	5	2	 |
01010100	0.5382	 | 	01010100	7	2	 |
00001000	0.5384	 | 	00001000	7	6	 |
10001110	0.5386	 | 	10001110	11	12	 |
11101111	0.5386	 | 	11101111	7	6	 |
00001110	0.5388	 | 	00001110	7	2	 |
01000101	0.5389	 | 	01000101	7	2	 |
11001110	0.5389	 | 	11001110	7	2	 |
11000000	0.5390	 | 	11000000	5	2	 |
11010100	0.5390	 | 	11010100	11	12	 |
11110011	0.5391	 | 	11110011	5	2	 |
10101111	0.5393	 | 	10101111	5	2	 |
01001100	0.5395	 | 	01001100	7	2	 |
01110111	0.5395	 | 	01110111	5	2	 |
11101000	0.5396	 | 	11101000	11	12	 |
10100010	0.5396	 | 	10100010	7	2	 |
10110000	0.5398	 | 	10110000	7	2	 |
01110001	0.5398	 | 	01110001	11	12	 |
11110111	0.5399	 | 	11110111	7	6	 |
10110010	0.5400	 | 	10110010	11	12	 |
01001111	0.5404	 | 	01001111	7	2	 |
00110011	0.5405	 | 	00110011	3	1	 |
10001010	0.5407	 | 	10001010	7	2	 |
11110010	0.5408	 | 	11110010	7	2	 |
01010001	0.5408	 | 	01010001	7	2	 |
10101000	0.5410	 | 	10101000	7	2	 |
01011101	0.5411	 | 	01011101	7	2	 |
00001101	0.5414	 | 	00001101	7	2	 |
11101010	0.5415	 | 	11101010	7	2	 |
00000010	0.5415	 | 	00000010	7	6	 |
10101110	0.5415	 | 	10101110	7	2	 |
11011100	0.5416	 | 	11011100	7	2	 |
00100010	0.5420	 | 	00100010	5	2	 |
11111000	0.5423	 | 	11111000	7	2	 |
01110000	0.5423	 | 	01110000	7	2	 |
10111010	0.5424	 | 	10111010	7	2	 |
10001100	0.5424	 | 	10001100	7	2	 |
11100000	0.5425	 | 	11100000	7	2	 |
01110101	0.5425	 | 	01110101	7	2	 |
11001101	0.5430	 | 	11001101	7	2	 |
00110010	0.5434	 | 	00110010	7	2	 |
00000011	0.5435	 | 	00000011	5	2	 |
11010000	0.5435	 | 	11010000	7	2	 |
00000001	0.5436	 | 	00000001	7	6	 |
11110100	0.5437	 | 	11110100	7	2	 |
11110001	0.5439	 | 	11110001	7	2	 |
11001000	0.5440	 | 	11001000	7	2	 |
11111011	0.5440	 | 	11111011	7	6	 |
00010111	0.5443	 | 	00010111	11	12	 |
11101100	0.5444	 | 	11101100	7	2	 |
11010101	0.5445	 | 	11010101	7	2	 |
00010101	0.5446	 | 	00010101	7	2	 |
01010111	0.5446	 | 	01010111	7	2	 |
10111011	0.5446	 | 	10111011	5	2	 |
01001110	0.5447	 | 	01001110	7	1	 |
00101010	0.5447	 | 	00101010	7	2	 |
11000100	0.5447	 | 	11000100	7	2	 |
11011110	0.5448	 | 	11011110	9	2	 |
10001111	0.5451	 | 	10001111	7	2	 |
01111111	0.5451	 | 	01111111	7	6	 |
10111110	0.5452	 | 	10111110	9	2	 |
10111111	0.5452	 | 	10111111	7	6	 |
00101110	0.5453	 | 	00101110	7	1	 |
00000111	0.5453	 | 	00000111	7	2	 |
01010110	0.5454	 | 	01010110	11	8	 |
11110110	0.5454	 | 	11110110	9	2	 |
01000110	0.5457	 | 	01000110	9	4	 |
10010000	0.5458	 | 	10010000	9	2	 |
01110010	0.5459	 | 	01110010	7	1	 |
10110110	0.5459	 | 	10110110	13	12	 |
11011010	0.5460	 | 	11011010	9	4	 |
01110011	0.5462	 | 	01110011	7	2	 |
00101011	0.5462	 | 	00101011	11	12	 |
00010010	0.5462	 | 	00010010	9	2	 |
00110100	0.5462	 | 	00110100	9	4	 |
00111111	0.5463	 | 	00111111	5	2	 |
01011110	0.5464	 | 	01011110	9	4	 |
11001010	0.5464	 | 	11001010	7	1	 |
00110001	0.5464	 | 	00110001	7	2	 |
10011010	0.5465	 | 	10011010	11	8	 |
01101100	0.5465	 | 	01101100	11	8	 |
11101101	0.5465	 | 	11101101	9	2	 |
11101011	0.5466	 | 	11101011	9	2	 |
00100011	0.5466	 | 	00100011	7	2	 |
00010100	0.5467	 | 	00010100	9	2	 |
01011100	0.5468	 | 	01011100	7	1	 |
10100111	0.5469	 | 	10100111	9	4	 |
01000111	0.5469	 | 	01000111	7	1	 |
01001000	0.5469	 | 	01001000	9	2	 |
01001001	0.5469	 | 	01001001	13	12	 |
00011110	0.5469	 | 	00011110	11	8	 |
00100001	0.5469	 | 	00100001	9	2	 |
01100001	0.5469	 | 	01100001	13	12	 |
00011111	0.5470	 | 	00011111	7	2	 |
00001011	0.5471	 | 	00001011	7	2	 |
10011101	0.5471	 | 	10011101	9	4	 |
10101001	0.5471	 | 	10101001	11	8	 |
11010010	0.5471	 | 	11010010	11	8	 |
00111011	0.5471	 | 	00111011	7	2	 |
10111101	0.5472	 | 	10111101	11	12	 |
11100010	0.5472	 | 	11100010	7	1	 |
01100000	0.5472	 | 	01100000	9	2	 |
00011011	0.5472	 | 	00011011	7	1	 |
10010011	0.5472	 | 	10010011	11	8	 |
10001101	0.5473	 | 	10001101	7	1	 |
11100011	0.5473	 | 	11100011	9	4	 |
11000011	0.5473	 | 	11000011	7	2	 |
11000010	0.5473	 | 	11000010	9	4	 |
01000001	0.5473	 | 	01000001	9	2	 |
00101100	0.5474	 | 	00101100	9	4	 |
10011100	0.5474	 | 	10011100	11	8	 |
10110011	0.5474	 | 	10110011	7	2	 |
01010010	0.5475	 | 	01010010	9	4	 |
10010110	0.5475	 | 	10010110	15	12	 |
11011011	0.5475	 | 	11011011	11	12	 |
01100100	0.5475	 | 	01100100	9	4	 |
11101001	0.5475	 | 	11101001	13	12	 |
01001010	0.5476	 | 	01001010	9	4	 |
01001011	0.5476	 | 	01001011	11	8	 |
01100101	0.5476	 | 	01100101	11	8	 |
00010110	0.5476	 | 	00010110	13	12	 |
11100100	0.5476	 | 	11100100	7	1	 |
11010110	0.5477	 | 	11010110	13	12	 |
10101011	0.5477	 | 	10101011	7	2	 |
01000011	0.5477	 | 	01000011	9	4	 |
01000010	0.5477	 | 	01000010	11	12	 |
00011100	0.5477	 | 	00011100	9	4	 |
01111011	0.5477	 | 	01111011	9	2	 |
10000001	0.5478	 | 	10000001	11	12	 |
01100010	0.5478	 | 	01100010	9	4	 |
10100001	0.5478	 | 	10100001	9	4	 |
10010010	0.5478	 | 	10010010	13	12	 |
01101010	0.5478	 | 	01101010	11	8	 |
11100101	0.5479	 | 	11100101	9	4	 |
10110111	0.5479	 | 	10110111	9	2	 |
00110101	0.5479	 | 	00110101	7	1	 |
00011000	0.5479	 | 	00011000	11	12	 |
01011011	0.5479	 | 	01011011	9	4	 |
01011001	0.5479	 | 	01011001	11	8	 |
00111000	0.5479	 | 	00111000	9	4	 |
00101000	0.5479	 | 	00101000	9	2	 |
00111001	0.5479	 | 	00111001	11	8	 |
00101001	0.5479	 | 	00101001	13	12	 |
10000101	0.5479	 | 	10000101	9	4	 |
01111000	0.5479	 | 	01111000	11	8	 |
10101101	0.5480	 | 	10101101	9	4	 |
11100001	0.5480	 | 	11100001	11	8	 |
10011111	0.5480	 | 	10011111	9	2	 |
01011000	0.5480	 | 	01011000	9	4	 |
01111010	0.5480	 | 	01111010	9	4	 |
01011010	0.5481	 | 	01011010	7	2	 |
10000100	0.5481	 | 	10000100	9	2	 |
10110100	0.5481	 | 	10110100	11	8	 |
11010111	0.5481	 | 	11010111	9	2	 |
11000111	0.5481	 | 	11000111	9	4	 |
10111100	0.5481	 | 	10111100	9	4	 |
10010100	0.5481	 | 	10010100	13	12	 |
11000001	0.5482	 | 	11000001	9	4	 |
01111100	0.5482	 | 	01111100	9	4	 |
00011010	0.5482	 | 	00011010	9	4	 |
11100111	0.5483	 | 	11100111	11	12	 |
01010011	0.5483	 | 	01010011	7	1	 |
01110100	0.5483	 | 	01110100	7	1	 |
00011001	0.5484	 | 	00011001	9	4	 |
11100110	0.5484	 | 	11100110	9	4	 |
00011101	0.5485	 | 	00011101	7	1	 |
01100111	0.5485	 | 	01100111	9	4	 |
10111000	0.5485	 | 	10111000	7	1	 |
10011000	0.5485	 | 	10011000	9	4	 |
11111001	0.5485	 | 	11111001	9	2	 |
00100111	0.5486	 | 	00100111	7	1	 |
00111110	0.5486	 | 	00111110	9	4	 |
10100110	0.5486	 | 	10100110	11	8	 |
10100011	0.5487	 | 	10100011	7	1	 |
00101111	0.5487	 | 	00101111	7	2	 |
11010011	0.5487	 | 	11010011	9	4	 |
10100100	0.5487	 | 	10100100	9	4	 |
11000110	0.5487	 | 	11000110	11	8	 |
00010011	0.5487	 | 	00010011	7	2	 |
11011000	0.5487	 | 	11011000	7	1	 |
10000010	0.5488	 | 	10000010	9	2	 |
00100110	0.5488	 | 	00100110	9	4	 |
10011110	0.5488	 | 	10011110	13	12	 |
10011001	0.5488	 | 	10011001	7	2	 |
10011011	0.5488	 | 	10011011	9	4	 |
00110110	0.5488	 | 	00110110	11	8	 |
01110110	0.5489	 | 	01110110	9	4	 |
10100101	0.5489	 | 	10100101	7	2	 |
01111001	0.5489	 | 	01111001	13	12	 |
00000110	0.5489	 | 	00000110	9	2	 |
10101100	0.5490	 | 	10101100	7	1	 |
11001001	0.5490	 | 	11001001	11	8	 |
01100110	0.5490	 | 	01100110	7	2	 |
10001011	0.5491	 | 	10001011	7	1	 |
00001001	0.5491	 | 	00001001	9	2	 |
01101111	0.5491	 | 	01101111	9	2	 |
01101101	0.5491	 | 	01101101	13	12	 |
00100101	0.5492	 | 	00100101	9	4	 |
01111101	0.5492	 | 	01111101	9	2	 |
11001011	0.5492	 | 	11001011	9	4	 |
01101000	0.5493	 | 	01101000	13	12	 |
01101001	0.5493	 | 	01101001	15	12	 |
00111101	0.5493	 | 	00111101	9	4	 |
10000111	0.5493	 | 	10000111	11	8	 |
00100100	0.5494	 | 	00100100	11	12	 |
11011001	0.5494	 | 	11011001	9	4	 |
10110101	0.5494	 | 	10110101	9	4	 |
01101110	0.5495	 | 	01101110	9	4	 |
11000101	0.5496	 | 	11000101	7	1	 |
00110111	0.5496	 | 	00110111	7	2	 |
10111001	0.5498	 | 	10111001	9	4	 |
00111100	0.5499	 | 	00111100	7	2	 |
10110001	0.5499	 | 	10110001	7	1	 |
01100011	0.5500	 | 	01100011	11	8	 |
01101011	0.5502	 | 	01101011	13	12	 |
00111010	0.5502	 | 	00111010	7	1	 |
01111110	0.5503	 | 	01111110	11	12	 |
10001001	0.5504	 | 	10001001	9	4	 |
10010101	0.5505	 | 	10010101	11	8	 |
10010111	0.5505	 | 	10010111	13	12	 |
00101101	0.5510	 | 	00101101	11	8	 |
10000110	0.5516	 | 	10000110	13	12	 |
11010001	0.5517	 | 	11010001	7	1	 |
10010001	0.5521	 | 	10010001	9	4	 |
10000011	0.5534	 | 	10000011	9	4	 |


00000000	0.0783	 | 	00000000	1	1	 |
11111111	0.1648	 | 	11111111	1	1	 |
11110000	0.3097	 | 	11110000	3	1	 |
11001100	0.3618	 | 	11001100	3	1	 |
10100000	0.3672	 | 	10100000	5	2	 |
10101010	0.3754	 | 	10101010	3	1	 |
11000000	0.3817	 | 	11000000	5	2	 |
10001000	0.4005	 | 	10001000	5	2	 |
10000000	0.4010	 | 	10000000	7	6	 |
00001111	0.4014	 | 	00001111	3	1	 |
01010101	0.4042	 | 	01010101	3	1	 |
00110011	0.4088	 | 	00110011	3	1	 |
01000100	0.4241	 | 	01000100	5	2	 |
11101000	0.4258	 | 	11101000	11	12	 |
00001010	0.4258	 | 	00001010	5	2	 |
00110000	0.4288	 | 	00110000	5	2	 |
01010000	0.4318	 | 	01010000	5	2	 |
00010000	0.4429	 | 	00010000	7	6	 |
00000011	0.4434	 | 	00000011	5	2	 |
00000101	0.4437	 | 	00000101	5	2	 |
10110010	0.4480	 | 	10110010	11	12	 |
00001100	0.4485	 | 	00001100	5	2	 |
00000100	0.4508	 | 	00000100	7	6	 |
00000001	0.4522	 | 	00000001	7	6	 |
00001000	0.4536	 | 	00001000	7	6	 |
01000000	0.4574	 | 	01000000	7	6	 |
11010100	0.4577	 | 	11010100	11	12	 |
00100000	0.4578	 | 	00100000	7	6	 |
11010000	0.4618	 | 	11010000	7	2	 |
11111010	0.4627	 | 	11111010	5	2	 |
00111111	0.4643	 | 	00111111	5	2	 |
00101011	0.4690	 | 	00101011	11	12	 |
10110000	0.4707	 | 	10110000	7	2	 |
00010001	0.4714	 | 	00010001	5	2	 |
10101000	0.4757	 | 	10101000	7	2	 |
11001000	0.4762	 | 	11001000	7	2	 |
11100000	0.4787	 | 	11100000	7	2	 |
00100010	0.4798	 | 	00100010	5	2	 |
01110000	0.4799	 | 	01110000	7	2	 |
01001101	0.4841	 | 	01001101	11	12	 |
10101111	0.4844	 | 	10101111	5	2	 |
01001100	0.4849	 | 	01001100	7	2	 |
01010100	0.4855	 | 	01010100	7	2	 |
00010111	0.4875	 | 	00010111	11	12	 |
01011111	0.4876	 | 	01011111	5	2	 |
00011111	0.4877	 | 	00011111	7	2	 |
11110010	0.4878	 | 	11110010	7	2	 |
11000100	0.4886	 | 	11000100	7	2	 |
10001100	0.4887	 | 	10001100	7	2	 |
11111100	0.4888	 | 	11111100	5	2	 |
00000010	0.4899	 | 	00000010	7	6	 |
01010001	0.4949	 | 	01010001	7	2	 |
10001110	0.4960	 | 	10001110	11	12	 |
00110111	0.4976	 | 	00110111	7	2	 |
10111111	0.5009	 | 	10111111	7	6	 |
11110100	0.5026	 | 	11110100	7	2	 |
10111000	0.5061	 | 	10111000	7	1	 |
10001010	0.5076	 | 	10001010	7	2	 |
00101111	0.5084	 | 	00101111	7	2	 |
11111000	0.5093	 | 	11111000	7	2	 |
01011101	0.5095	 | 	01011101	7	2	 |
11001111	0.5104	 | 	11001111	5	2	 |
01000101	0.5104	 | 	01000101	7	2	 |
10101011	0.5106	 | 	10101011	7	2	 |
10111011	0.5114	 | 	10111011	5	2	 |
00010101	0.5133	 | 	00010101	7	2	 |
00111011	0.5136	 | 	00111011	7	2	 |
01110111	0.5142	 | 	01110111	5	2	 |
10101110	0.5142	 | 	10101110	7	2	 |
01010111	0.5149	 | 	01010111	7	2	 |
10011100	0.5165	 | 	10011100	11	8	 |
10111010	0.5176	 | 	10111010	7	2	 |
01001111	0.5177	 | 	01001111	7	2	 |
10100010	0.5188	 | 	10100010	7	2	 |
10001111	0.5195	 | 	10001111	7	2	 |
11111101	0.5210	 | 	11111101	7	6	 |
00001101	0.5218	 | 	00001101	7	2	 |
00110101	0.5222	 | 	00110101	7	1	 |
11101110	0.5225	 | 	11101110	5	2	 |
11001101	0.5237	 | 	11001101	7	2	 |
01110001	0.5251	 | 	01110001	11	12	 |
10111100	0.5252	 | 	10111100	9	4	 |
11010101	0.5252	 | 	11010101	7	2	 |
01111111	0.5253	 | 	01111111	7	6	 |
11110101	0.5260	 | 	11110101	5	2	 |
10011111	0.5263	 | 	10011111	9	2	 |
00110010	0.5264	 | 	00110010	7	2	 |
00101010	0.5268	 | 	00101010	7	2	 |
10101100	0.5275	 | 	10101100	7	1	 |
10110001	0.5276	 | 	10110001	7	1	 |
00011100	0.5280	 | 	00011100	9	4	 |
00111101	0.5283	 | 	00111101	9	4	 |
11011100	0.5287	 | 	11011100	7	2	 |
00101100	0.5289	 | 	00101100	9	4	 |
01101100	0.5289	 | 	01101100	11	8	 |
11110011	0.5290	 | 	11110011	5	2	 |
11011111	0.5303	 | 	11011111	7	6	 |
00001011	0.5307	 | 	00001011	7	2	 |
10011000	0.5315	 | 	10011000	9	4	 |
01110101	0.5319	 | 	01110101	7	2	 |
01111011	0.5321	 | 	01111011	9	2	 |
00010011	0.5327	 | 	00010011	7	2	 |
10110011	0.5329	 | 	10110011	7	2	 |
11100010	0.5332	 | 	11100010	7	1	 |
00100011	0.5333	 | 	00100011	7	2	 |
01110011	0.5339	 | 	01110011	7	2	 |
11000001	0.5341	 | 	11000001	9	4	 |
00011011	0.5358	 | 	00011011	7	1	 |
11001010	0.5358	 | 	11001010	7	1	 |
11000110	0.5363	 | 	11000110	11	8	 |
11011101	0.5364	 | 	11011101	5	2	 |
00100111	0.5369	 | 	00100111	7	1	 |
10001101	0.5376	 | 	10001101	7	1	 |
10011011	0.5377	 | 	10011011	9	4	 |
11001110	0.5378	 | 	11001110	7	2	 |
11101100	0.5379	 | 	11101100	7	2	 |
10010111	0.5379	 | 	10010111	13	12	 |
11101111	0.5380	 | 	11101111	7	6	 |
10010000	0.5386	 | 	10010000	9	2	 |
10010010	0.5391	 | 	10010010	13	12	 |
00111100	0.5391	 | 	00111100	7	2	 |
01111100	0.5391	 | 	01111100	9	4	 |
10010011	0.5393	 | 	10010011	11	8	 |
11010011	0.5393	 | 	11010011	9	4	 |
11010001	0.5393	 | 	11010001	7	1	 |
10010001	0.5399	 | 	10010001	9	4	 |
10001011	0.5407	 | 	10001011	7	1	 |
11000111	0.5408	 | 	11000111	9	4	 |
01110100	0.5411	 | 	01110100	7	1	 |
00110100	0.5411	 | 	00110100	9	4	 |
10000101	0.5412	 | 	10000101	9	4	 |
01100000	0.5412	 | 	01100000	9	2	 |
00110001	0.5415	 | 	00110001	7	2	 |
01001000	0.5422	 | 	01001000	9	2	 |
01101111	0.5423	 | 	01101111	9	2	 |
10111001	0.5425	 | 	10111001	9	4	 |
11111001	0.5425	 | 	11111001	9	2	 |
10010100	0.5426	 | 	10010100	13	12	 |
11100111	0.5427	 | 	11100111	11	12	 |
11000101	0.5427	 | 	11000101	7	1	 |
11100001	0.5429	 | 	11100001	11	8	 |
01010011	0.5430	 | 	01010011	7	1	 |
11000011	0.5437	 | 	11000011	7	2	 |
11100110	0.5437	 | 	11100110	9	4	 |
11001011	0.5438	 | 	11001011	9	4	 |
11101010	0.5441	 | 	11101010	7	2	 |
00000111	0.5442	 | 	00000111	7	2	 |
01000111	0.5450	 | 	01000111	7	1	 |
01000011	0.5450	 | 	01000011	9	4	 |
00100101	0.5453	 | 	00100101	9	4	 |
01100101	0.5453	 | 	01100101	11	8	 |
10101101	0.5453	 | 	10101101	9	4	 |
11101101	0.5453	 | 	11101101	9	2	 |
01001011	0.5454	 | 	01001011	11	8	 |
10011110	0.5456	 | 	10011110	13	12	 |
01101011	0.5457	 | 	01101011	13	12	 |
00011000	0.5459	 | 	00011000	11	12	 |
00111000	0.5461	 | 	00111000	9	4	 |
01111000	0.5461	 | 	01111000	11	8	 |
01001110	0.5462	 | 	01001110	7	1	 |
00111010	0.5462	 | 	00111010	7	1	 |
11011000	0.5464	 | 	11011000	7	1	 |
11011011	0.5465	 | 	11011011	11	12	 |
10000100	0.5474	 | 	10000100	9	2	 |
00100100	0.5474	 | 	00100100	11	12	 |
10110111	0.5475	 | 	10110111	9	2	 |
11100100	0.5482	 | 	11100100	7	1	 |
01100100	0.5486	 | 	01100100	9	4	 |
10101001	0.5492	 | 	10101001	11	8	 |
11101001	0.5492	 | 	11101001	13	12	 |
10110100	0.5492	 | 	10110100	11	8	 |
10110101	0.5495	 | 	10110101	9	4	 |
11110001	0.5499	 | 	11110001	7	2	 |
10100011	0.5499	 | 	10100011	7	1	 |
10000011	0.5499	 | 	10000011	9	4	 |
10111110	0.5504	 | 	10111110	9	2	 |
01100011	0.5505	 | 	01100011	11	8	 |
10111101	0.5513	 | 	10111101	11	12	 |
11010111	0.5515	 | 	11010111	9	2	 |
00011101	0.5522	 | 	00011101	7	1	 |
10000111	0.5528	 | 	10000111	11	8	 |
11110111	0.5529	 | 	11110111	7	6	 |
10011010	0.5534	 | 	10011010	11	8	 |
01011011	0.5542	 | 	01011011	9	4	 |
11011110	0.5544	 | 	11011110	9	2	 |
11101011	0.5545	 | 	11101011	9	2	 |
01011110	0.5547	 | 	01011110	9	4	 |
01111110	0.5547	 | 	01111110	11	12	 |
00010010	0.5548	 | 	00010010	9	2	 |
11011010	0.5550	 | 	11011010	9	4	 |
00100001	0.5553	 | 	00100001	9	2	 |
00101110	0.5554	 | 	00101110	7	1	 |
00010100	0.5555	 | 	00010100	9	2	 |
01100111	0.5555	 | 	01100111	9	4	 |
01100110	0.5561	 | 	01100110	7	2	 |
11100011	0.5571	 | 	11100011	9	4	 |
00111110	0.5574	 | 	00111110	9	4	 |
10100100	0.5575	 | 	10100100	9	4	 |
01011100	0.5576	 | 	01011100	7	1	 |
00001001	0.5578	 | 	00001001	9	2	 |
01000001	0.5591	 | 	01000001	9	2	 |
00011001	0.5610	 | 	00011001	9	4	 |
01011001	0.5610	 | 	01011001	11	8	 |
01110010	0.5611	 | 	01110010	7	1	 |
01001001	0.5619	 | 	01001001	13	12	 |
01101110	0.5622	 | 	01101110	9	4	 |
11111110	0.5626	 | 	11111110	7	6	 |
10000010	0.5627	 | 	10000010	9	2	 |
01100001	0.5634	 | 	01100001	13	12	 |
01111101	0.5639	 | 	01111101	9	2	 |
10010101	0.5643	 | 	10010101	11	8	 |
01101000	0.5646	 | 	01101000	13	12	 |
10011001	0.5650	 | 	10011001	7	2	 |
10001001	0.5650	 | 	10001001	9	4	 |
11011001	0.5650	 | 	11011001	9	4	 |
11001001	0.5650	 | 	11001001	11	8	 |
10000110	0.5665	 | 	10000110	13	12	 |
10011101	0.5668	 | 	10011101	9	4	 |
01001010	0.5670	 | 	01001010	9	4	 |
11111011	0.5672	 | 	11111011	7	6	 |
10000001	0.5673	 | 	10000001	11	12	 |
11010010	0.5687	 | 	11010010	11	8	 |
10100001	0.5688	 | 	10100001	9	4	 |
01101010	0.5691	 | 	01101010	11	8	 |
01000010	0.5692	 | 	01000010	11	12	 |
01100010	0.5694	 | 	01100010	9	4	 |
11110110	0.5700	 | 	11110110	9	2	 |
01110110	0.5700	 | 	01110110	9	4	 |
01010110	0.5700	 | 	01010110	11	8	 |
00110110	0.5700	 | 	00110110	11	8	 |
10110110	0.5700	 | 	10110110	13	12	 |
11010110	0.5700	 | 	11010110	13	12	 |
00010110	0.5700	 | 	00010110	13	12	 |
10010110	0.5700	 | 	10010110	15	12	 |
00001110	0.5700	 | 	00001110	7	2	 |
00101101	0.5705	 | 	00101101	11	8	 |
01101101	0.5705	 | 	01101101	13	12	 |
01010010	0.5720	 | 	01010010	9	4	 |
00111001	0.5721	 | 	00111001	11	8	 |
00101001	0.5721	 | 	00101001	13	12	 |
01111001	0.5721	 | 	01111001	13	12	 |
01101001	0.5721	 | 	01101001	15	12	 |
10100101	0.5724	 | 	10100101	7	2	 |
11100101	0.5724	 | 	11100101	9	4	 |
00011010	0.5728	 | 	00011010	9	4	 |
01011010	0.5736	 | 	01011010	7	2	 |
01000110	0.5746	 | 	01000110	9	4	 |
01011000	0.5750	 | 	01011000	9	4	 |
00100110	0.5750	 | 	00100110	9	4	 |
10100110	0.5750	 | 	10100110	11	8	 |
00000110	0.5762	 | 	00000110	9	2	 |
10100111	0.5766	 | 	10100111	9	4	 |
11000010	0.5809	 | 	11000010	9	4	 |
00011110	0.5843	 | 	00011110	11	8	 |
01111010	0.5861	 | 	01111010	9	4	 |
00101000	0.5864	 | 	00101000	9	2	 |

 */