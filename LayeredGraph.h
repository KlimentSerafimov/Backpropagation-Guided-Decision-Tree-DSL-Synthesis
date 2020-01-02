//
// Created by Kliment Serafimov on 2019-02-16.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_LAYERED_GRAPH_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_LAYERED_GRAPH_H

#include "Layer.h"

class LayeredGraph
{
public:
    LayeredGraph(){};
    
    //Graph new_graph;
    
    LayeredGraph(LayeredGraph const &_leyered_graph)
    {
        layers = _leyered_graph.layers;
    }
    
    vector<Layer> layers;
    
    void perturb(double rate)
    {
        for(int i = 0;i<layers.size();i++)
        {
            layers[i].perturb(rate);
        }
    }
    
    int size()
    {
        return (int)layers.size();
    }
    
    int count_backprop;
    int count_feedforward;
    
    int numInputs;
    int numOutputs;
    
    void constructNN(int *hiddenLayer)
    {
        int neuronsPerLayer[20] = {numInputs};
        int c = 1;
        while(hiddenLayer[c-1]!=-1)
        {
            neuronsPerLayer[c] = hiddenLayer[c-1];
            c++;
        }
        
        neuronsPerLayer[c] = numOutputs;
        neuronsPerLayer[c+1] = -1;
        
        int at_layer = 0;
        while(neuronsPerLayer[at_layer+1]!=-1)
        {
            //assert(neuronsPerLayer[at_layer+1] != 3);
            layers.push_back(Layer(neuronsPerLayer[at_layer], neuronsPerLayer[at_layer+1]));
            at_layer++;
        }
    }
    
    void constructNN(int *hiddenLayer, vector<bit_signature> all_w)
    {
        int neuronsPerLayer[20] = {numInputs};
        int c = 1;
        while(hiddenLayer[c-1]!=-1)
        {
            neuronsPerLayer[c] = hiddenLayer[c-1];
            c++;
        }
        
        neuronsPerLayer[c] = numOutputs;
        neuronsPerLayer[c+1] = -1;
        
        int at_layer = 0;
        while(neuronsPerLayer[at_layer+1]!=-1)
        {
            //assert(neuronsPerLayer[at_layer+1] != 3);
            
            layers.push_back(Layer(neuronsPerLayer[at_layer], neuronsPerLayer[at_layer+1]));
            at_layer++;
        }
        int all_w_idx = 0;
        for(int i = 0;i<layers.size();i++)
        {
            for(int j = 0;j<layers[i].neurons.size();j++)
            {
                for(int k = 0;k<layers[i].neurons[j].weights.size();k++)
                {
                    layers[i].neurons[j].weights[k].value = all_w[all_w_idx++].value;
                }
                layers[i].neurons[j].t = all_w[all_w_idx++].value;
            }
        }
        
        assert(all_w_idx == all_w.size());
        
    }
    
    void single_construct_neuron(vector<int> special)
    {
        layers.push_back(Layer(numInputs, 1));
        for(int i = 0;i<special.size();i++)
        {
            layers[0].disregard_input(special[i]);
        }
        int next_layer_num_inputs = 1;
        int main_id = 0;
        vector<vector<int> > new_age_neurons;
        for(int i = 0;i<(1<<special.size());i++)
        {
            vector<int> subset;
            for(int j = 0;j<special.size();j++)
            {
                if((i&(1<<j))!=0)
                {
                    subset.push_back(special[j]);
                }
            }
            vector<int> v_pair;
            v_pair.push_back(main_id);
            v_pair.push_back(layers[0].add_neuron_with_inputs(special));
            next_layer_num_inputs++;
            new_age_neurons.push_back(v_pair);
            
        }
        layers.push_back(Layer(next_layer_num_inputs, 0));
        for(int i = 0;i<new_age_neurons.size();i++)
        {
            layers[1].add_neuron_with_inputs(new_age_neurons[i]);
        }
        layers.push_back(Layer((int)new_age_neurons.size(), numOutputs));
    }
    
    vector<bit_signature> * forwardPropagate(vector<bit_signature> *input, bool remember)
    {
        count_feedforward+=remember;
        
        vector<bit_signature>* layerOutput = input;
        for(int j=0; j<layers.size(); j++)
        {
            layerOutput = layers[j].output(layerOutput, remember);
        }
        return layerOutput;
    }
    
    void backwardPropagate(vector<bit_signature> desiredOutput, vector<bit_signature> predicted, double rate, bool apply)
    {
        count_backprop++;
        
        
        vector<bit_signature> Derivatives;
        Derivatives = vector<bit_signature>(desiredOutput.size(), 0.0);
        
        double sum = 0;
        for(int j=0; j<desiredOutput.size(); j++)
        {
            double difference = (desiredOutput[j]-predicted[j]);
            sum+=difference*difference;
        
            Derivatives[j]+=difference;
        }
        for(int j=layers.size()-1; j>=0; j--)
        {
            //cout << "Layer " << j <<": ";
            /*for(int i = 0;i<Derivatives.size();i++)
            {
                cout << Derivatives[i] <<" ";
            }
            cout << endl;*/
            Derivatives = layers[j].updateWeights(Derivatives, rate, apply);
           /* for(int i = 0;i<layers[j].sum_abs_delta_der.size();i++)
            {
                cout << layers[j].sum_abs_delta_der[i] <<" ";
            }
            cout << endl;*/
        }
        /*for(int i = 0;i<Derivatives.size();i++)
        {
            cout << Derivatives[i] <<" ";
        }
        cout << endl;
        cout << endl;*/
    }
    
    void batchApplyDeltaWeights()
    {
        for(int j=(int)layers.size()-1; j>=0; j--)
        {
            layers[j].batchApplyDeltaWeights();
        }
    }
    
    void printBrainStructure()
    {
        cout << layers.size() <<" Layers" << endl;
        for(int i=0;i<layers.size();i++)
        {
            cout << layers[i].neurons.size() <<" ";
        }
        cout << endl;
    }
    
    void printWeights()
    {
        cout << endl;
        cout << "Network = " <<endl;
        cout << "{"<<endl;
        
        for(int i=0; i<layers.size(); i++)
        {
            if(i != 0)
            {
                cout << ", " <<endl;
            }
            layers[i].printWeights();
        }
        cout << endl << "}" <<endl;
    }
    
    LayeredGraph copy()
    {
        LayeredGraph ret = LayeredGraph();
        ret.layers = layers;
        return ret;
    }
};

#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_LAYERED_GRAPH_H
