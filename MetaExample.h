//
// Created by Kliment Serafimov on 3/7/20.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_METAEXAMPLE_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_METAEXAMPLE_H
//
// Created by Kliment Serafimov on 2019-11-18.
//

#include "PartialFunction.h"
#include <string>

using namespace std;

class MetaExample
{
public:
    PartialFunction partial_function;
    PartialFunction generalization;
    int idx;
    bool active = true;

    MetaExample() = default;

    MetaExample(PartialFunction partial_function, PartialFunction total_function, int idx);
    MetaExample(int _function_size, Bitvector _total_function, Bitvector _partition);
    MetaExample(int _function_size, Bitvector _total_function, Bitvector _partition, Bitvector generalization_partition);
    MetaExample(int _function_size, Bitvector _total_function, Bitvector _partition, Bitvector generalization_partition, int _idx);
    MetaExample(int _function_size, Bitvector _total_function, Bitvector _partition, Bitvector generalization_partition, int _idx, bool _active);
    MetaExample(PartialFunction partial_function);

    int get_compact_partial_outputs();
    int get_compact_hidden_outputs();

    string linear_string(int tab);

    MetaExample get_application_of_subdomain(Bitvector subdomain_mask)
    {
        return MetaExample(
                partial_function.function_size,
                generalization.total_function & subdomain_mask,
                partial_function.partition & subdomain_mask,
                generalization.partition & subdomain_mask,
                idx,
                active
        );
    }

    bool fully_constrained()
    {
        return (partial_function.partition & generalization.partition) == generalization.partition;
    }

    string to_string();

    void print();

    bool operator < (const MetaExample &other) const
    {
        return generalization.total_function < other.generalization.total_function;
    }

    bool is_consistent_with(MetaExample other);

    int get_function_size()
    {
        assert(partial_function.function_size == generalization.function_size);
        return partial_function.function_size;
    }

    bool next_mask(Bitvector &v, int num_bits)// current permutation of bits
    {
        int count = 0;
        int function_size = get_function_size();
        for(int i = 0;i<function_size;i++)
        {
            if(v.test(i))
            {
                if(i == function_size - 1)
                {
                    return false;
                }
                else
                {
                    v.set(i, 0);
                    if(!v.test(i+1))
                    {
                        v.set(i+1);
                        for(int j = 0;j<count;j++)
                        {
                            v.set(j);
                        }
                        break;
                    }
                    else
                    {
                        count++;
                    }
                }
            }
        }
        return true;
    }

//    vector<Bitvector> get_masks(int max_mask_size);

};

int get_num_missing_bits(vector<MetaExample> meta_examples);

#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_METAEXAMPLE_H
