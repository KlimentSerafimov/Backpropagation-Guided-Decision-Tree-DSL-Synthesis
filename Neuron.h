//
// Created by Kliment Serafimov on 2019-02-16.
//


#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_NEURON_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_NEURON_H

#include "Header.h"
#include "bit_signature.h"


class Neuron
{
public:

    //long long K = 1, S = 3;

    double sigmoid_derivative(double alpha);
    double sigmoid(double x);

    int num_inputs;

    double previous_output;
    double t;

    pair<int, double> cumulative_delta_t;

    vector<bit_signature> previous_input;
    vector<bit_signature> weights;

    vector<bit_signature> abs_delta_der;

    vector<pair<int, double> > cumulative_delta_w;

    vector<bool> disregard;

    void perturb(double rate);

    bit_signature get_weight(int id);

    int rate = 1;

    double get_random_weight();
    double get_random_weight(double lb, double hb);
    void set_random_weight();

    void minus(Neuron other);
    void mul(double alpha);

    Neuron(int _num_in);
    void add_input(int n);
    void disregard_input(int input_id);
    double output(vector<bit_signature> input, bool remember);
    vector<bit_signature> update_weights(double prevDer, double rate, bool apply);
    string printWeights();
    void batchApplyDeltaWeights();

};

#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_NEURON_H
