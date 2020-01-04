
/* Code written by Kliment Serafimov */

#include "Neuron.h"
#include "Layer.h"
#include "Data.h"
#include "Header.h"
#include "bit_signature.h"
#include "LayeredGraph.h"
#include "Batch.h"
#include "NeuralNetwork.h"
#include "DecisionTree.h"
#include "FirstOrderLearning.h"
#include "Policy.h"
#include "DecisionTreeSynthesisViaDP.h"
#include "FirstOrderDataset.h"
#include "SecondOrderDataset.h"
#include "SecondOrderLearning.h"
#include "FunctionAndDecisionTreeScore.h"
#include "print_all_decision_tree_strings.h"
#include "archaic_neural_designed_circuit.h"
#include "dataset_initializers.h"
#include "BitvectorFunctionSolver.h"

ofstream fout("table.out");

ofstream fout_experiment("experiment.out");

vector<vector<pair<pair<double, double>, DecisionTreeScore> > > get_ensamble_neural_error_buckets
        (int n, FirstOrderDataset<DataAndDecisionTreeScore> first_order_data, NeuralNetwork::parameters params)
{
    vector<vector<pair<pair<double, double>, DecisionTreeScore> > > ret;

    vector<DataAndDecisionTreeScore> sorted;

    vector<NeuralNetworkAndScore*> ensamble_nets = params.progressive_ensamble_nets[n];

    int leaf_iter = params.get_iteration_count(n);

    sorted = first_order_data.train_data;

    sort_v(sorted);

    first_order_data.train_data = sorted;


    for(int i = 0; i < ensamble_nets.size(); i++)
    {
        cout << "HERE " << ensamble_nets[i]->print() << endl;
        vector<double> sums;
        vector<int> count;
        bool prev_score_defined = false;
        DecisionTreeScore prev_score;

        //cout << "size = " << first_order_data.train_data.size() <<endl;


        vector<pair<pair<double, double>, DecisionTreeScore> > overlapping;

        cout << "first_order_data.train_data.size() = " << first_order_data.train_data.size() <<endl;

        for(int j = 0; j < first_order_data.train_data.size();j++)
        {


            NeuralNetwork leaf_learner = ensamble_nets[i];
//            NeuralNetwork::parameters leaf_parameters = cutoff_param(leaf_iter, 0.01);
            params.set_iteration_count(leaf_iter);
            leaf_learner.train(&first_order_data.train_data[j], params, params.priority_train_f);
            double error = leaf_learner.get_error_of_data(&first_order_data.train_data[j]).second;

            cout << first_order_data.train_data[j].printConcatinateOutput() << "\t" << first_order_data.train_data[j].score.size << "\t" << error << endl;

            if(prev_score_defined)
            {
                if(prev_score.size == first_order_data.train_data[j].score.size)
                {
                    int last_id = sums.size()-1;
                    overlapping[last_id].first.first = min(error, overlapping[last_id].first.first);
                    overlapping[last_id].first.second = max(error, overlapping[last_id].first.second);
                    sums[last_id] += error;
                    count[last_id] += 1;
                }
                else
                {
                    overlapping.push_back(make_pair(make_pair(error, error), first_order_data.train_data[j].score));
                    sums.push_back(error);
                    count.push_back(1);
                    prev_score = first_order_data.train_data[j].score;
                }
            }
            else
            {
                overlapping.push_back(make_pair(make_pair(error, error), first_order_data.train_data[j].score));
                sums.push_back(error);
                count.push_back(1);
                prev_score = first_order_data.train_data[j].score;
                prev_score_defined = true;
            }

        }

        assert(sums.size() == count.size());

        cout <<"n = " << n << " sums.size() == " << sums.size() <<endl;


        for(int local_i = 0;local_i<(int)sums.size()-1;local_i++)
        {
            assert(local_i<sums.size()-1);
//            cout << local_i << " "<< sums.size()-1 << endl;
            cout << "local_i = " << local_i <<endl;
            cout << "sums[local_i] = " << sums[local_i]<<" | count[local_i] = " << count[local_i] <<" | ";
            double at_now = sums[local_i]/count[local_i];
            double at_next = sums[local_i]/count[local_i];
            double mid = (at_now+at_next)/2;
            if(overlapping[local_i].first.second < overlapping[local_i+1].first.first)
            {
                //all good
            }
            else
            {
                overlapping[local_i].first.second = mid;
                overlapping[local_i + 1].first.first = mid;
            }
            cout << "at_now = " << at_now <<" | " << "at_next = " << at_next <<" | mid = " << mid << endl;
        }

        overlapping[sums.size()-1].first.second = 1;

        ret.push_back(overlapping);

    }

    cout << "MODEL BRACKETS:" <<endl;

    for(int i = 0;i<ret.size();i++)
    {
        for(int j = 0;j<ret[i].size();j++)

        {
            cout << ret[i][j].first.first <<" "<< ret[i][j].first.second <<" "<< ret[i][j].second.size <<" | " <<endl;
        }
    }


    return ret;
}

vector<int> hidden_layer_block(int width, int depth)
{
    vector<int> ret;
    for(int i = 0;i<depth;i++)
    {
        ret.push_back(width);
    }
    return ret;
}

vector<vector<vector<pair<pair<double, double>, DecisionTreeScore> > > > get_progressive_ensamble_neural_error_buckets
        (int n, vector<FirstOrderDataset<DataAndDecisionTreeScore> > first_order_datasets, NeuralNetwork::parameters params)
{

    vector<vector<vector<pair<pair<double, double>, DecisionTreeScore> > > > ret;
    ret.push_back(vector<vector<pair<pair<double, double>, DecisionTreeScore> > > ());
    for(int i = 1; i <= n; i++)
    {
        if(first_order_datasets[i].train_data.size()>=1) {
            ret.push_back(get_ensamble_neural_error_buckets
                                  (i, first_order_datasets[i], params));
        } else{
            ret.push_back(vector<vector<pair<pair<double, double>, DecisionTreeScore> > >());
        }
    }
    return ret;
}

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
                    (init_initialization, first_order_dataset.train_data, param, false, true);

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

    /*

    vector<
        pair<
            vector<DataType> train_data,
            vector<DataType> test_data
            >
        > train_data;

    vector<
        pair<
            vector<DataType> train_data,
            vector<DataType> test_data
            >
        > test_data;

     */

};

vector<DataAndDecisionTreeScore> select_all_functions(int n)
{
    assert(n<=3);

    vector<DataAndDecisionTreeScore> ret;

    vector<FunctionAndDecisionTreeScore> sample_ids = get_smallest_f(n);

    if(n == 3)
    {
        srand(0);
        for(int i = 0;i<sample_ids.size();i++)
        {
            if(sample_ids[i].size < 7)
            {
                DataAndDecisionTreeScore new_data(sample_ids[i]);
                new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].function);

                ret.push_back(new_data);
            }
            else if(/*sample_ids[i].size <= 9 && */__builtin_popcount(i) <= 2)
            {
                DataAndDecisionTreeScore new_data(sample_ids[i]);
                new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].function);

                ret.push_back(new_data);
            }
        }
    }
    else
    {
        assert(0);
    }

    return ret;
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

    int num_samples = 4;
    holy_grail.test_data = sample_partitions(first_order_dataset.test_data, num_samples);

    int num_repeats = 3;

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

                param.leaf_iter_init = 18;
                param.max_boundary_size = 5;
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
                    cout << "NOW TRAINING ON FIRST_ORDER_DATASET.TRAIN[" << first_order_dataset_id << "]" << endl;

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

class latentDecisionTreeExtractor: public SecondOrderLearning<Data>
{
public:

    latentDecisionTreeExtractor(): SecondOrderLearning()
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
        cout << "IN latentDecisionTreeExtractor" <<endl <<endl;

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

                vector<int> hidden_layer_widths = hidden_layer_block(hidden_layer_width[local_n - 1], hidden_layer_depth[local_n-1]);

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

            bool do_learn = false;

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
//            if(do_learn)
//            {
//                cutoff_parameter.progressive_ensamble_neural_error_buckets =
//                        get_progressive_ensamble_neural_error_buckets
//                                (local_n - 1, first_order_datasets, cutoff_parameter);
//            }

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

void see_delta_w()
{
    const int n = 3;

    const int middle_layer = n;

    NeuralNetwork walker = NeuralNetwork(n, middle_layer, 1);

    vector<pair<NeuralNetwork::PriorityTrainReport, int> > to_sort;

    vector<NeuralNetwork> net_data;

    Data meta_task;

    NeuralNetwork walker_init = NeuralNetwork(n, middle_layer, 1);

    for(int i = 0; i<(1<<(1<<n))/8;i++)
    {
        Data the_data;
        the_data.init_exaustive_table_with_unary_output(n, i);

        NeuralNetwork::parameters param = NeuralNetwork::parameters(1, 1);//rate, batch_width

        walker.set_special_weights();
        //walker = walker_init;
        walker.save_weights();
        printOnlyBatch = false;
        cout << bitset<(1<<n)>(i).to_string()<<endl;
        param.track_dimension_model = false;
        param.accuracy = 0.25/2;
        to_sort.push_back(make_pair(walker.train(&the_data, param, &NeuralNetwork::softPriorityTrain), i));

        NeuralNetwork::test_score score = walker.test(&the_data, 0.25/2);

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

    NeuralNetwork net_learning_nets = NeuralNetwork(noramlized_meta_task.numInputs, 2*noramlized_meta_task.numInputs, noramlized_meta_task.numOutputs);

    NeuralNetwork::parameters param = NeuralNetwork::parameters(1, 3);//rate, batch_width
    printItteration = true;
    param.track_dimension_model = false;
    param.accuracy = 0.0125*2;
    net_learning_nets.train(&noramlized_meta_task, param, &NeuralNetwork::softPriorityTrain);

    Data resulting_data;

    int total_error = 0;
    for(int i = 0; i<noramlized_meta_task.size();i++)
    {
        vector<bit_signature> bit_signature_vector = to_bit_signature_vector((1<<n), i);
        vector<bit_signature> network_output = net_learning_nets.forwardPropagate(&bit_signature_vector, false);

        noramlized_meta_task.unnormalize(network_output);

        NeuralNetwork output_network = NeuralNetwork(n, middle_layer, 1, network_output);
        //output_network.printWeights();

        Data the_data;
        the_data.init_exaustive_table_with_unary_output(n, i);

        NeuralNetwork::test_score score = output_network.test(&the_data, 0.5);
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

        NeuralNetwork::parameters param = NeuralNetwork::parameters(1, 1);//rate, batch_width

        walker.set_special_weights();
        //walker.save_weights();
        printOnlyBatch = true;
        cout << bitset<(1<<n)>(i).to_string()<<" ";
        to_sort.push_back(make_pair(walker.train(&the_data, param, &NeuralNetwork::softPriorityTrain), i));
        //walker.compare_to_save();
    }
    cout << endl;
    sort_v(to_sort);
    for(int i = 0;i<to_sort.size();i++)
    {
        cout << bitset<(1<<n)>(to_sort[i].second).to_string() <<endl;
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
                latentDecisionTreeExtractor trainer = latentDecisionTreeExtractor();

                trainer.train(3);
            }
            if(false)
            {
                multi_task_learning_for_neural_error_ordering();
            }
            if(true)
            {
                int min_trainable_n = 2;
                int max_trainable_n = 4;

                vector<FirstOrderDataset<DataAndDecisionTreeScore> > datasets;
                for(int i = 0;i<min_trainable_n-1;i++)
                {
                    datasets.push_back(FirstOrderDataset<DataAndDecisionTreeScore>());//n < min_trainable_n
                }

                for(int i = min_trainable_n-1; i<=max_trainable_n; i++)
                {
                    datasets.push_back(init_custom_data_and_difficulty_set(i, 100, 60));
                }

                for(int num_iter = 20; num_iter <= 100 ; num_iter += 40)
                {
                    vector<int> leaf_iters;
                    leaf_iters.push_back(-1);//no leaf_iter for n = 0;
                    leaf_iters.push_back(8);// for n = 1;
                    leaf_iters.push_back(16);// for n = 2;
                    leaf_iters.push_back(18);// for n = 3;
                    leaf_iters.push_back(32);// for n = 4;
                    leaf_iters.push_back(64);// for n = 5;
                    leaf_iters.push_back(128);// for n = 4;
                    leaf_iters.push_back(256);// for n = 5;

                    vector<int> hidden_layer_width;
                    hidden_layer_width.push_back(-1);//no leaf_iter for n = 0;
                    hidden_layer_width.push_back(4);// for n = 1;
                    hidden_layer_width.push_back(6);// for n = 2;
                    hidden_layer_width.push_back(8);// for n = 3;
                    hidden_layer_width.push_back(10);// for n = 4;
                    hidden_layer_width.push_back(12);// for n = 5;


                    vector<int> hidden_layer_depth;
                    hidden_layer_depth.push_back(-1);//no leaf_iter for n = 0;
                    hidden_layer_depth.push_back(1);// for n = 1;
                    hidden_layer_depth.push_back(1);// for n = 2;
                    hidden_layer_depth.push_back(1);// for n = 3;
                    hidden_layer_depth.push_back(1);// for n = 4;
                    hidden_layer_depth.push_back(1);// for n = 5;

                    for(int repeat = 0; repeat < 1; repeat++) {

//                            assert(min_trainable_n == max_trainable_n);
                        NeuralNetwork ensamble_progressive_nets[10][10];

                        typename FirstOrderLearning<DataAndDecisionTreeScore>::learning_parameters all_params =
                                FirstOrderLearning<DataAndDecisionTreeScore>::learning_parameters();
                        all_params.at_initiation_of_parameters_may_21st_10_52_pm();

                        all_params.leaf_iter_init = leaf_iters[3];
                        all_params.treshold = 0.4;
                        all_params.max_boundary_size = 1;
                        all_params.max_num_iter = num_iter;

                        /*int leaf_iter_init;

                        double treshold;*/

                        int num_ensambles = 1;

                        fout_experiment << num_iter <<"\t";

//                            get_local_ensamble<DataAndDecisionTreeScore>
//                                    (min_trainable_n-1, hidden_layer_width, hidden_layer_depth,
//                                     , datasets, ensamble_progressive_nets, all_params);


                        latentDecisionTreeExtractor SpicyAmphibian = latentDecisionTreeExtractor();

//                            assert(false);

                        SpicyAmphibian.train_library<DataAndDecisionTreeScore>
                                (min_trainable_n, max_trainable_n, leaf_iters, hidden_layer_width, hidden_layer_depth,
                                 num_ensambles, datasets, all_params);
                    }
                }
                return;
            }
            if(false)
            {
                cout << "See delta_w" << endl;
                see_delta_w();
                return;
            }
            if(false)
            {
                DecisionTreeSynthesisViaDP<Data> dper;
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
                NeuralNetwork::parameters param = NeuralNetwork::parameters(1.5, 3);
                param.num_stale_iterations = 10000;//7;
                param.set_iteration_count(16);//(16);
                param.ensamble_size = 1;//8;
                param.priority_train_f = &NeuralNetwork::softPriorityTrain;
                param.decision_tree_synthesiser_type = confusion_guided;

                DecisionTree tree = DecisionTree(&the_data, param);

                return ;
            }

            if(false) //via circuit
            {
                NeuralNetwork::parameters param = NeuralNetwork::parameters(1.5, 3);

                archaic_neural_designed_circuit two_nets;
                param.num_stale_iterations = 7;
                param.set_iteration_count(16);

                string type = "longestSubstring_ai_is_1";
                Data the_data = Data();
                int n = 4, outputSize;
                the_data.generateData(n, outputSize, type);

                two_nets.build_circuit_per_output_dimension(the_data, param);
                //two_nets.build_circuit_based_on_singletons(6, param, &NeuralNetwork::softPriorityTrain);
                return;
            }


            if(false)
            {
                //single init root/local build
                archaic_neural_decision_tree tree = archaic_neural_decision_tree();

                if(false) {
                    tree.local_single_build(4, NeuralNetwork::parameters(1, 2));
                    tree.print_gate(0);
                }
                if(true) {

                    NeuralNetwork::parameters param = NeuralNetwork::parameters(3.79, 3);
                    //NeuralNetwork::parameters param = NeuralNetwork::parameters(1, 3);

                    param.num_stale_iterations = 3;
                    int to_print = tree.init_root(4, param, &NeuralNetwork::softPriorityTrain);

                    cout << to_print << endl;

                    int num = tree.print_gate(0);

                    cout << "num gates = " << num << endl;

                    cout << endl;
                }
                return;
            }



            if(false) {
                print_discrete_model = false;
                printItteration = false;
                printNewChildren = false;

                //for(double id = 0.5;id<=10;id+=0.5)
                for (int id_iter_count = 16; id_iter_count <= 16; id_iter_count *= 2) {
                    cout << endl;
                    cout << id_iter_count << " ::: " << endl;
                    for (int id_ensamble_size = 1; id_ensamble_size <= 12; id_ensamble_size++) {
                        cout << id_ensamble_size << "\t" << " :: " << "\t";
                        for (int i = 0; i < 6; i++) {
                            NeuralNetwork::parameters param = NeuralNetwork::parameters(1.5, 3);
                            param.num_stale_iterations = 7;
                            param.set_iteration_count(id_iter_count);
                            param.ensamble_size = id_ensamble_size;
                            param.priority_train_f = &NeuralNetwork::softPriorityTrain;
                            param.decision_tree_synthesiser_type = confusion_guided;


                            string type = "longestSubstring_ai_is_1";
                            Data the_data = Data();
                            int n = 6, outputSize;
                            the_data.generateData(n, outputSize, type);

                            if(true) {
                                DecisionTree tree = DecisionTree(&the_data, param);
                                cout << tree.size << "\t";
                            }
                            if(false) {
                                archaic_neural_decision_tree tree = archaic_neural_decision_tree();
                                cout << endl;
                                cout << tree.init_root(6, param, &NeuralNetwork::softPriorityTrain) <<"\t";

                                cout << endl;
                                int num = tree.print_gate(0);

                                cout << "num gates = " << num <<endl;
                                cout << endl <<endl;
                            }

                        }
                        cout << endl;
                    }
                }
                return;
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
             int num_nodes = tree.init_root(8, learning_rate, 1, &NeuralNetwork::queueTrain);
             cout  << num_nodes << "\t";
             }
             cout << endl;
             }*/
        }
    }
};

int main()
{

//    print_all_decision_tree_strings(3);
//    return 0;

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
