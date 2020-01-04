//
// Created by Kliment Serafimov on 2020-01-02.
//

#include "util.h"

vector<bit_signature> to_bit_signature_vector(int num_bits, int bit)
{
    vector<bit_signature> ret;
    for(int i = 0;i<num_bits;i++)
    {
        ret.push_back((double)((bit&(1<<i))!=0));
        ret[i].vector_id = i;
    }
    return ret;
}

double sum_vector(vector<bit_signature> v)
{
    double ret = 0;
    for(int i = 0;i<v.size();i++)
    {
        ret+=v[i];
    }
    return ret;
}

double max_of_vector(vector<bit_signature> v)
{
    double ret = 0;
    for(int i = 0;i<v.size();i++)
    {
        ret = max(ret, (double)v[i]);
    }
    return ret;
}

double get_max_of_error_vector(vector<bit_signature>* correct, vector<bit_signature> predict, int the_pow)
{
    assert(correct->size()==predict.size());
    double max_error = -1;
    for(int i=0; i<predict.size(); i++)
    {
        double difference = correct->at(i)-predict.at(i);
        max_error = max(max_error, abs(pow(difference, the_pow)));
    }
    return max_error;
}

vector<bit_signature> get_error_vector(vector<bit_signature> correct, vector<bit_signature> predict, int the_pow)
{
    assert(correct.size()==predict.size());
    vector<bit_signature> error;
    for(int i=0; i<predict.size(); i++)
    {
        double difference = correct[i]-predict[i];;
        error.push_back(abs(pow(difference, the_pow)));
    }
    return error;
}

bool check(vector<bit_signature> *correct, vector<bit_signature> predict, double accuracy)
{
    assert(correct->size()==predict.size());
    for(int i=0; i<predict.size(); i++)
    {
        //assert(((predict[i]>0.5)!=(correct[i]>0.5)) == (abs(correct[i]-predict[i])>0.5));
        //if((predict[i]>0.5)!=(correct[i]>0.5))
        if(abs(correct->at(i)-predict.at(i))>accuracy)
        {
            return false;
        }
    }
    return true;
}

vector<bit_signature> vector_value_product(double value, vector<bit_signature> vec)
{
    for(int i = 0;i<vec.size();i++)
    {
        vec[i]*=value;
    }
    return vec;
}

vector<bit_signature> pairwise_negation(vector<bit_signature> first, vector<bit_signature> second)
{
    assert(first.size() == second.size());
    for(int i = 0;i<first.size();i++)
    {
        first[i] -= second[i];
    }
    return first;
}

vector<bit_signature> pairwise_addition(vector<bit_signature> first, vector<bit_signature> second)
{
    assert(first.size() == second.size());
    for(int i = 0;i<first.size();i++)
    {
        first[i] += second[i];
    }
    return first;
}

vector<bit_signature> pairwise_product(vector<bit_signature> first, vector<bit_signature> second)
{
    assert(first.size() == second.size());
    for(int i = 0;i<first.size();i++)
    {
        first[i] *= second[i];
    }
    return first;
}

vector<bit_signature> pairwise_division(vector<bit_signature> first, vector<bit_signature> second)
{
    assert(first.size() == second.size());
    for(int i = 0;i<first.size();i++)
    {
        first[i] /= second[i];
    }
    return first;
}

void cout_vector(vector<int> v)
{
    for(int i=0;i<v.size();i++)
    {
        cout << v[i] <<" ";
    }
    cout << endl;
}

string toString(int n, int k, int l)
{
    string ret = "";
    int init_n = n;
    while(n>0)
    {
        ret+=((n%k)+'0');
        n/=k;
    }

    if(l >= 0)
    {
        while (ret.size() < l)
        {
            ret += "0";
        }
    }

    rev_v(ret);
    return ret;
}

string toBinaryString(int n, int l)
{
    return toString(n, 2, l);
}
string toDecimalString(int n)
{
    return toString(n, 10, -1);
}

string indent(int n)
{
    string s = "    ";
    string ret = "";
    for(int i = 0;i<n;i++)
    {
        ret+=s;
    }
    return ret;
}


string get_tab(int num_tabs)
{
    string ret;
    for(int i = 0;i<num_tabs;i++)
    {
        ret+="\t";
    }
    return ret;
}