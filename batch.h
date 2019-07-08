//
// Created by Kliment Serafimov on 2019-02-16.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_BATCH_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_BATCH_H

class batch_element
{
public:
    int id;
    double representative_error;
    double error;
    vector<bit_signature> error_vector;
    vector<bit_signature> network_output;
    
    int iter = -1;
    
    batch_element()
    {
        
    }
    
    batch_element(int _id)
    {
        id = _id;
    }
    
    batch_element(double _error, int _id)
    {
        error = representative_error = _error;
        id = _id;
    }
    
    bool operator < (const batch_element &other) const
    {
        return representative_error < other.representative_error;
    }
};

class batch //ids
{
public:
    
    vector<batch_element> elements;
    
    double error_sum = 0;
    double error_average = 0;
    
    vector<bit_signature> vector_error_sum;
    vector<bit_signature> vector_error_average;
    
    vector<bit_signature> network_output_sum;
    vector<bit_signature> network_output_average;
    
    //seperatey used
    vector<bit_signature> input_sum;
    vector<bit_signature> input_average;
    
    void set_inputs(Data* latice)
    {
        assert(input_sum.size() == 0);
        assert(input_average.size() == 0);
        input_sum = vector<bit_signature>(latice->numInputs, 0);
        input_average = vector<bit_signature>(latice->numInputs, 0);
        
        for(int i = 0;i<input_sum.size(); i++)
        {
            for(int j = 0;j<size();j++)
            {
                input_sum[i] += latice->in[elements[j].id][i];
            }
            input_average[i] = input_sum[i]/size();
        }
    }
    
    int size()
    {
        return (int)elements.size();
    }
    
    void init_vectors(batch_element new_element)
    {
        vector_error_sum.resize(new_element.error_vector.size(), 0);
        vector_error_average.resize(new_element.error_vector.size(), 0);
        network_output_sum.resize(new_element.network_output.size(), 0);
        network_output_average.resize(new_element.network_output.size(), 0);
    }
    
    void push_back(batch_element new_element)
    {
        elements.push_back(new_element);
        error_sum+=new_element.error;
        error_average = error_sum/size();
        
        
        if(vector_error_sum.size() == 0)
        {
            init_vectors(new_element);
        }
        
        assert(vector_error_sum.size() == new_element.error_vector.size());
        assert(vector_error_average.size() == new_element.error_vector.size());
        for(int i = 0;i<vector_error_sum.size();i++)
        {
            vector_error_sum[i] += new_element.error_vector[i];
            vector_error_average[i] = vector_error_sum[i]/size();
        }
        
        assert(network_output_sum.size() == new_element.network_output.size());
        assert(network_output_average.size() == new_element.network_output.size());
        for(int i = 0;i<network_output_sum.size();i++)
        {
            network_output_sum[i] += new_element.network_output[i];
            network_output_average[i] = network_output_sum[i]/size();
        }
        
    }
    
};


#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_BATCH_H
