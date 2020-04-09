//
// Created by Kliment Serafimov on 2020-01-05.
//

#include "see_delta_w.h"

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