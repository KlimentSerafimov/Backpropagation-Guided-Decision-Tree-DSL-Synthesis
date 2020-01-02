//
// Created by Kliment Serafimov on 2019-02-16.
//


#include "Neuron.h"

double Neuron::sigmoid_derivative(double alpha) { /*K++;*/ return alpha*(1-alpha); }

double Neuron::sigmoid(double x) { /*S++;*/ return 1/(1+exp(-x)); }


double Neuron::get_random_weight()
{
    double lb = -0.5, hb = 0.5;
    return (hb-lb)*(double)rand(0, RAND_MAX-1)/(RAND_MAX-1)+lb;
}

double Neuron::get_random_weight(double lb, double hb)
{
    return (hb-lb)*(double)rand(0, RAND_MAX-1)/(RAND_MAX-1)+lb;
}

void Neuron::set_random_weight()
{
    t = get_random_weight();
    for(int i = 0;i<weights.size();i++)
    {
        weights[i] = get_random_weight();
    }
}

Neuron::Neuron(int _num_in)
{
    num_inputs = _num_in;
    t = get_random_weight();
    cumulative_delta_t = make_pair(0, 0);
    for(int i=0; i<num_inputs; i++)
    {

        weights.push_back(get_random_weight());
        cumulative_delta_w.push_back(make_pair(0, 0));
        abs_delta_der.push_back(0.0);
        disregard.push_back(false);

    }
}


void Neuron::add_input(int n)
{
    num_inputs+=n;
    for(int i=0; i<n; i++)
    {
        weights.push_back(get_random_weight());
        cumulative_delta_w.push_back(make_pair(0, 0));
        abs_delta_der.push_back(0.0);
    }
}

void Neuron::disregard_input(int input_id)
{
    assert(disregard[input_id] == false);
    disregard[input_id] = true;
}

double Neuron::output(vector<bit_signature> *input, bool remember)
{
    assert(input->size()==num_inputs);

    double sum = -t;
    if(remember)previous_input.clear();
    for(int j=0; j<num_inputs; j++)
    {
        if(!disregard[j])
        {
            if(remember)previous_input.push_back(input->at(j));
            sum += input->at(j)*weights[j];
        }
    }
    double ret = sigmoid(sum);
    if(remember)previous_output = ret;
    assert(ret >= 0);
    return ret;
}
vector<bit_signature> Neuron::update_weights(double prevDer, double rate, bool apply)
{
    vector<bit_signature> ret;
    for(int i=0; i<num_inputs; i++)
    {
        if(!disregard[i])
        {
            double next_delta_der = prevDer* sigmoid_derivative(previous_output)*weights[i];
            ret.push_back(next_delta_der);
            double delta_w = rate*prevDer* sigmoid_derivative(previous_output)*previous_input[i];

            abs_delta_der[i] = abs(next_delta_der);


            if(apply)
            {
                assert(cumulative_delta_w[i].first == 0);
                weights[i]+=delta_w;
            }
            else
            {
                cumulative_delta_w[i].first ++;
                cumulative_delta_w[i].second += delta_w;
            }
        }
    }

    double delta_t = rate*prevDer* sigmoid_derivative(previous_output)*(-1);
    if(apply)
    {
        assert(cumulative_delta_t.first == 0);
        t+=delta_t;
    }
    else
    {
        cumulative_delta_t.first ++;
        cumulative_delta_t.second += delta_t;
    }
    return ret;
}


void Neuron::batchApplyDeltaWeights()
{
    for(int i=0; i<num_inputs; i++)
    {
        if(!disregard[i])
        {
            double delta_w = cumulative_delta_w[i].second/(double)cumulative_delta_w[i].first;
            weights[i]+=delta_w;

            cumulative_delta_w[i].second = cumulative_delta_w[i].first = 0;
        }
    }

    double delta_t = cumulative_delta_t.second/(double)cumulative_delta_t.first;
    t+=delta_t;

    cumulative_delta_t.second = cumulative_delta_t.first = 0;
}

string Neuron::printWeights()
{
    string ret  = "{ {";
    for(int i=0; i<num_inputs; i++)
    {
        if(i!= 0)
        {
            ret+= ", ";
        }
        ret += to_string(weights[i]*(!disregard[i]));
    }
    ret+="}, {" +to_string(t) + "} }";
    return ret;
}


void Neuron::minus(Neuron other)
{
    assert(weights.size() == other.weights.size());
    for(int i = 0;i<weights.size();i++)
    {
        //cout << weights[i] << " - "<< other.weights[i] <<" = ";
        weights[i] = weights[i] - other.weights[i];
        //cout << weights[i] <<" other = " << other.weights[i] << endl;
    }
}

void Neuron::mul(double alpha)
{
    t *= alpha;
    for(int i = 0;i<weights.size();i++)
    {
        weights[i] *= alpha;
    }
}

void Neuron::perturb(double rate)
{
    t+= get_random_weight(-rate, rate);
    for(int i = 0;i<weights.size();i++)
    {
        weights[i]+= get_random_weight(-rate, rate);
    }
}

bit_signature Neuron::get_weight(int id)
{
    assert(0<=id && id<weights.size());
    return weights[id]*(!disregard[id]);
}