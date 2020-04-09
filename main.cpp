
/* Code written by Kliment Serafimov */

#include "print_all_decision_tree_strings.h"
#include "archaic_neural_designed_circuit.h"
#include "BitvectorFunctionSolver.h"
#include "multi_task_learning_for_neural_error_ordering.h"
#include "LatentDecisionTreeExtractor.h"
#include "see_delta_w.h"


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
                //construct exaustive dataset
                //implement contracting reptile
            }
            if(false)
            {
                print_all_decision_tree_strings(3);
            }
            if(false)
            {
                LatentDecisionTreeExtractor trainer = LatentDecisionTreeExtractor();
                trainer.train(3);
            }
            if(true)
            {
                multi_task_learning_for_neural_error_ordering();
            }
            if(false)
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


                        LatentDecisionTreeExtractor SpicyAmphibian = LatentDecisionTreeExtractor();

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
