//
// Created by Kliment Serafimov on 2019-02-16.
//


#include "Header.h"
#include "bit_signature.h"

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_NEURON_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_NEURON_H


class neuron
{
public:

    //long long K = 1, S = 3;

    double Kliment(double alpha);
    double Serafimov(double x);

    int num_in;

    double prevOutput;
    double t;

    pair<int, double> cumulative_delta_t;

    vector<bit_signature> previn;
    vector<bit_signature> w;

    vector<bit_signature> abs_delta_der;

    vector<pair<int, double> > cumulative_delta_w;

    vector<bool> disregard;

    void perturb(double rate)
    {
        t+=get_random_w(-rate, rate);
        for(int i = 0;i<w.size();i++)
        {
            w[i]+=get_random_w(-rate, rate);
        }
    }

    bit_signature get_weight(int id)
    {
        assert(0<=id && id<w.size());
        return w[id]*(!disregard[id]);
    }

    int rate = 1;

    double get_random_w();
    double get_random_w(double lb, double hb);
    void set_random_w();
    void minus(neuron other)
    {
        assert(w.size() == other.w.size());
        for(int i = 0;i<w.size();i++)
        {
            //cout << w[i] << " - "<< other.w[i] <<" = ";
            w[i] = w[i] - other.w[i];
            //cout << w[i] <<" other = " << other.w[i] << endl;
        }
    }
    void mul(double alpha)
    {
        t *= alpha;
        for(int i = 0;i<w.size();i++)
        {
            w[i] *= alpha;
        }
    }
    neuron(int _num_in);
    void addInput(int n);
    void disregardInput(int input_id);

    double output(vector<bit_signature> input, bool remember);
    vector<bit_signature> updateWeights(double prevDer, double rate, bool apply);
    string printWeights();
    void batchApplyDeltaWeights();

};


#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_NEURON_H
