//
// Created by Kliment Serafimov on 2019-11-18.
//

#include "MetaExample.h"
#include "util.h"
#include <iostream>

MetaExample::MetaExample(
        int _function_size, Bitvector _total_function, Bitvector _partition) {
    partial_function = PartialFunction(_function_size, _total_function, _partition);
    generalization = PartialFunction(_function_size, _total_function, Bitvector((1<<_function_size)-1));
}

MetaExample::MetaExample(int _function_size, Bitvector _total_function, Bitvector _partition, Bitvector generalization_partition) {

    partial_function = PartialFunction(_function_size, _total_function, _partition);
    generalization = PartialFunction(_function_size, _total_function, generalization_partition);
}


MetaExample::MetaExample(PartialFunction _partial_function){
    partial_function = _partial_function;
    generalization = PartialFunction(
            partial_function.function_size, partial_function.total_function, -1);
}


int MetaExample::get_compact_partial_outputs()
{
    int compact_partial_outputs = 0;
    for(int i = 0, j = 0;i<partial_function.function_size; i++) {
        if (get_bit(partial_function.partition, i)) {
            compact_partial_outputs += (1 << j) * get_bit(partial_function.total_function, i);
            j++;
        }
    }
    return compact_partial_outputs;
}

int MetaExample::get_compact_hidden_outputs() {
    int compact_hidden_outputs = 0;
    for(int i = 0, j = 0;i<partial_function.function_size; i++) {
        if (!get_bit(partial_function.partition, i)) {
            compact_hidden_outputs += (1 << j) * get_bit(partial_function.total_function, i);
            j++;
        }
    }
    return compact_hidden_outputs;
}


string MetaExample::to_string() {
    return linear_string(0);
}

string MetaExample::linear_string(int tab)
{
    string ret;
    for(int i = 0;i<tab;i++)
    {
        ret+="\t";
    }
//    ret += "{";
//    ret += "(";
//    ret += "   ";
    ret += partial_function.to_string();
    ret += " -> ";
    ret += generalization.to_string();
//    ret += ")";
//    ret += "}";
    return ret;
}

void MetaExample::print()
{
    cout << to_string() << endl;
}

MetaExample::MetaExample(int _function_size, Bitvector _total_function, Bitvector _partition, Bitvector generalization_partition, int _idx) {
    partial_function = PartialFunction(_function_size, _total_function, _partition);
    generalization = PartialFunction(_function_size, _total_function, generalization_partition);
    idx = _idx;
}

MetaExample::MetaExample(int _function_size, Bitvector _total_function, Bitvector _partition, Bitvector generalization_partition, int _idx, bool _active) {
    partial_function = PartialFunction(_function_size, _total_function, _partition);
    generalization = PartialFunction(_function_size, _total_function, generalization_partition);
    idx = _idx;
    active = _active;
}

bool MetaExample::is_consistent_with(MetaExample other) {
    assert(generalization.partition == other.generalization.partition);

    if(other.generalization.is_contained_in(partial_function) &&
       generalization.is_contained_in(other.partial_function))
    {
        bool ret =
                (other.generalization.total_function & other.generalization.partition) ==
                (generalization.total_function & generalization.partition);
        return ret;
    }

    return true;
}

MetaExample::MetaExample(PartialFunction _partial_function, PartialFunction _generalization, int _idx) {
    assert(_generalization.is_contained_in(_partial_function));
    partial_function = _partial_function;
    generalization = _generalization;
    idx = _idx;
}

//vector<Bitvector> MetaExample::get_masks(int max_mask_size)
//{
//    vector<Bitvector> ret;
//    if(get_function_size() == 0)
//    {
//        return ret;
//    }
//    cout << "MASKS: " << endl;
//    for(int i = 1;i<=max_mask_size;i++)
//    {
//        Bitvector local_mask(all_zeroes, get_function_size());
//        for(int j = 0;j<i;j++)
//        {
//            local_mask.set(j);
//        }
//        do{
//            ret.push_back(local_mask);
//            cout << bitvector_to_str(local_mask, get_function_size()) << endl;
//        }while(next_mask(local_mask, i));
//    }
//    return ret;
//}

int get_num_missing_bits(vector<MetaExample> meta_examples)
{
    int ret = 0;
    for(int i = 0; i<meta_examples.size();i++)
    {
        ret += meta_examples[i].generalization.partition.count() -
               meta_examples[i].partial_function.partition.count();
    }
    return ret;
}


