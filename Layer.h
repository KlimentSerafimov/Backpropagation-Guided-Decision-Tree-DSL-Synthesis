//
// Created by Kliment Serafimov on 2019-02-16.
//


#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_LAYER_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_LAYER_H

#include "Header.h"
#include "Neuron.h"
#include "util.h"

class Layer
{
public:
    int num;
    int n_in;
    int n_out;

    vector<Neuron> neurons;
    vector<bit_signature> sum_abs_delta_der;

    void perturb(double rate)
    {
        for(int i = 0;i<neurons.size();i++)
        {
            neurons[i].perturb(rate);
        }
    }

    int size()
    {
        return (int)neurons.size();
    }

    bool invariant()
    {
        return num >= 1 && n_in >= 1;
    }

    Layer(int _in_per_neuron, int _num_neurons)
    {
        num = _num_neurons;
        n_in = _in_per_neuron;
        assert(invariant());
        //n_out = _out_per_neuron;
        for(int i=0; i<num; i++)
        {
            neurons.push_back(Neuron(n_in));
        }
        /*
         field_range in = field_range(_in_per_neuron, 0);
         field_range neurons = field_range(_num_neurons, out.real_end+1);
         field_range neurons = field_range(_num_neurons, out.real_end+1);
         field_range out = field_range(_out_per_neuron, in.real_end+1);

         for(field_element i = in; i<=in;++i)
         {
         for(field_element j = neurons; j<=neurons; ++j)
         {
         matrix[idx(i)][idx(j)] = 1;
         }
         }*/
        /*
         for(field_element i = neurons; i<=neurons; ++i)
         {
         for(field_element j = out; j<=out; ++j)
         {
         matrix[idx(i)][idx(j)] = 1;
         }
         }*/

    }
    void set_random_w()
    {
        for(int i = 0;i<neurons.size();i++)
        {
            neurons[i].set_random_weight();
        }
    }

    void minus(Layer other)
    {
        assert(neurons.size() == other.neurons.size());
        for(int i = 0;i<neurons.size();i++)
        {
            neurons[i].minus(other.neurons[i]);
        }
    }
    void mul(double alpha)
    {
        for(int i = 0;i<neurons.size();i++)
        {
            neurons[i].mul(alpha);
        }
    }

    void addOutput(int n)
    {
        num+=n;
        for(int i=0; i<n; i++)
        {
            neurons.push_back(Neuron(n_in));
        }
    }
    void addInput(int n)
    {
        n_in+=n;
        for(int i=0; i<num; i++)
        {
            neurons[i].add_input(n);
        }
    }
    void disregard_input(int input_id)
    {
        for(int i = 0;i<num;i++)
        {
            neurons[i].disregard_input(input_id);
        }
    }
    int add_neuron_with_inputs(vector<int> special)
    {
        sort_v(special);
        num++;
        Neuron new_neuron = Neuron(n_in);
        for(int i = 0, j = 0;i<n_in;i++)
        {
            if(special[i] != special[j])
            {
                new_neuron.disregard_input(i);
            }
            else
            {
                j++;
            }
        }
        int ret_id = (int)neurons.size();
        neurons.push_back(new_neuron);
        return ret_id;
    }
    vector<bit_signature> * output(vector<bit_signature> *input, bool remember)
    {
        vector<bit_signature>* layerOutput = new vector<bit_signature>();
        for(int i=0; i<num; i++)
        {
            layerOutput->push_back(neurons[i].output(input, remember));
        }
        return layerOutput;
    }
    vector<bit_signature> updateWeights(vector<bit_signature> derivatives, double rate, bool apply)
    {
        vector<bit_signature> sumDerivatives(n_in, 0);
        sum_abs_delta_der.clear();
        sum_abs_delta_der.resize(n_in, 0);
        for(int i=0; i<num; i++)
        {
            vector<bit_signature> oneDerivative = neurons[i].update_weights(derivatives[i], rate, apply);
            for(int j=0; j<n_in; j++)
            {
                sumDerivatives[j]+=oneDerivative[j];
                sum_abs_delta_der[j]+=neurons[i].abs_delta_der[j];
            }
        }
        return sumDerivatives;
    }
    void batchApplyDeltaWeights()
    {
        for(int i=0; i<num; i++)
        {
            neurons[i].batchApplyDeltaWeights();
        }
    }
    void printWeights()
    {
        cout << indent(1) << "{" <<endl;
        for(int i=0; i<num; i++)
        {
            if(i != 0)
            {
                cout << ", " <<endl;
            }
            cout << indent(2) << neurons[i].printWeights();
        }
        cout << endl << indent(1) << "}";
    }
};

#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_LAYER_H
