//
// Created by Kliment Serafimov on 2019-02-16.
//


#include "neuron.h"

double neuron::Kliment(double alpha) { /*K++;*/ return alpha*(1-alpha); }

double neuron::Serafimov(double x) { /*S++;*/ return 1/(1+exp(-x)); }


double neuron::get_random_w()
{
    double lb = -0.5, hb = 0.5;
    return (hb-lb)*(double)rand(0, RAND_MAX-1)/(RAND_MAX-1)+lb;
}

double neuron::get_random_w(double lb, double hb)
{
    return (hb-lb)*(double)rand(0, RAND_MAX-1)/(RAND_MAX-1)+lb;
}

void neuron::set_random_w()
{
    t = get_random_w();
    for(int i = 0;i<w.size();i++)
    {
        w[i] = get_random_w();
    }
}

neuron::neuron(int _num_in)
{
    num_in = _num_in;
    t = get_random_w();
    cumulative_delta_t = mp(0, 0);
    for(int i=0; i<num_in; i++)
    {

        w.pb(get_random_w());
        cumulative_delta_w.push_back(mp(0, 0));
        abs_delta_der.push_back(0.0);
        disregard.push_back(false);

    }
}


void neuron::addInput(int n)
{
    num_in+=n;
    for(int i=0; i<n; i++)
    {
        w.pb(get_random_w());
        cumulative_delta_w.push_back(mp(0, 0));
        abs_delta_der.push_back(0.0);
    }
}

void neuron::disregardInput(int input_id)
{
    assert(disregard[input_id] == false);
    disregard[input_id] = true;
}

double neuron::output(vector<bit_signature> input, bool remember)
{
    assert(input.size()==num_in);

    double sum = -t;
    if(remember)previn.clear();
    for(int j=0; j<num_in; j++)
    {
        if(!disregard[j])
        {
            if(remember)previn.pb(input[j]);
            sum += input[j]*w[j];
        }
    }
    double ret = Serafimov(sum);
    if(remember)prevOutput = ret;
    assert(ret >= 0);
    return ret;
}
vector<bit_signature> neuron::updateWeights(double prevDer, double rate, bool apply)
{
    vector<bit_signature> ret;
    for(int i=0; i<num_in; i++)
    {
        if(!disregard[i])
        {
            double next_delta_der = prevDer*Kliment(prevOutput)*w[i];
            ret.pb(next_delta_der);
            double delta_w = rate*prevDer*Kliment(prevOutput)*previn[i];

            abs_delta_der[i] = abs(next_delta_der);
            //abs_delta_der[i] = abs(prevDer*Kliment(prevOutput)*w[i]);


            if(apply)
            {
                assert(cumulative_delta_w[i].f == 0);
                w[i]+=delta_w;
            }
            else
            {
                cumulative_delta_w[i].f ++;
                cumulative_delta_w[i].s += delta_w;
            }
        }
    }
    //cout << endl;
    double delta_t = rate*prevDer*Kliment(prevOutput)*(-1);
    if(apply)
    {
        assert(cumulative_delta_t.f == 0);
        t+=delta_t;
    }
    else
    {
        cumulative_delta_t.f ++;
        cumulative_delta_t.s += delta_t;
    }
    return ret;
}


void neuron::batchApplyDeltaWeights()
{
    for(int i=0; i<num_in; i++)
    {
        if(!disregard[i])
        {
            double delta_w = cumulative_delta_w[i].s/(double)cumulative_delta_w[i].f;
            w[i]+=delta_w;

            cumulative_delta_w[i].s = cumulative_delta_w[i].f = 0;
        }
    }

    double delta_t = cumulative_delta_t.s/(double)cumulative_delta_t.f;
    t+=delta_t;

    cumulative_delta_t.s = cumulative_delta_t.f = 0;
}

string neuron::printWeights()
{
    string ret  = "{ {";
    for(int i=0; i<num_in; i++)
    {
        if(i!= 0)
        {
            ret+= ", ";
        }
        ret += to_string(w[i]*(!disregard[i]));
    }
    ret+="}, {" +to_string(t) + "} }";
    return ret;
}