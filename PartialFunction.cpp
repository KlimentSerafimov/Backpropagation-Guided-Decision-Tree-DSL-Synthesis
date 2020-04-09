//
// Created by Kliment Serafimov on 2019-11-18.
//

#include "PartialFunction.h"
#include <iostream>
#include "util.h"
#include <set>

PartialFunction::PartialFunction() = default;

PartialFunction::PartialFunction(int _function_size, Bitvector _total_function) {
    function_size = _function_size;
    total_function = _total_function;
    partition.set_range(0, function_size-1);
}

PartialFunction::PartialFunction(int _function_size, Bitvector _total_function, Bitvector _partition) {
    function_size = _function_size;
    total_function = _total_function;
    if(_partition == -1)
    {
        assert(false);
        _partition.set_range(0, function_size-1);
    }
    partition = _partition;
}

string PartialFunction::to_string() {
    string ret;
    for(int i = function_size-1;i>=0;i--)
    {
        if(get_bit(partition, i))
        {
            ret+='0'+get_bit(total_function, i);
        }
        else
        {
            ret+="_";
        }
    }
    return ret;
}

bool PartialFunction::is_contained_in(PartialFunction other_partial_function) {
    assert(function_size == other_partial_function.function_size);
    if((partition & other_partial_function.partition) == other_partial_function.partition) {
//    assert(partition == (1<<(1<<function_size))-1);
        bool ret = ((total_function & other_partial_function.partition) ==
                    (other_partial_function.total_function & other_partial_function.partition));
        return ret;
    } else
    {
        return false;
    }
}
//
//PartialFunction PartialFunction::get_composition(PartialFunction other) {
//    assert(function_size == other.function_size);
//
//    Bitvector common_partition = partition & other.partition;
//
//    assert((total_function & common_partition) == (other.total_function & common_partition));
//
//    Bitvector this_part = total_function & other.partition.get_flipped();
//    Bitvector other_part = other.total_function & partition.get_flipped();
//    Bitvector common_part = total_function & common_partition;
//
//    return PartialFunction(function_size, this_part | common_part | other_part, partition | other.partition);
//}

int PartialFunction::has_output(int idx) {
    return partition.test(idx);
}

int PartialFunction::get_output(int idx) {
    return total_function.test(idx);
}

void PartialFunction::apply_common_partition(PartialFunction other) {

    partition &= other.partition;
    partition &= ((total_function & partition) ^ (other.total_function & partition)).flip();

    assert((partition & total_function) == (partition & other.total_function));
}

int PartialFunction::partition_size() {
    return partition.count();
}

void PartialFunction::set_bit(int idx, int value) {
    if(!get_bit(partition, idx))
    {
        partition.set(idx);
    }
    total_function.set(idx, value);
    assert(get_bit(total_function, idx) == value);
}

void PartialFunction::clear_bit(int idx) {
    assert(get_bit(partition, idx) == 1);
    partition.set(idx, 0);
}

bool PartialFunction::empty() {
    return partition == 0;
}

//bool PartialFunction::has_empty_intersection_with(PartialFunction other) {
//    Bitvector common_partition = partition & other.partition;
//    Bitvector difference_mask = ((total_function & common_partition) ^ (other.total_function & common_partition));
//    return difference_mask != 0;
//}
//
//void PartialFunction::append_difference_with(PartialFunction other, vector<PartialFunction> &to_append_to) {
//
//    if (!has_empty_intersection_with(other)) {
//        Bitvector template_for_output__partition = partition;
//        Bitvector template_for_output__function = total_function & partition;
//
//        Bitvector other_contains_but_this_doesnt = other.partition ^ (other.partition & partition);
//
//        while(other_contains_but_this_doesnt != 0)
//        {
//            int idx_of_first_one = num_trailing_zeroes(other_contains_but_this_doesnt);
//
//            template_for_output__partition.set(idx_of_first_one);
//
//            Bitvector new_template_for_output__function = template_for_output__function;
//            new_template_for_output__function.set(idx_of_first_one, 1-get_bit(other.total_function, idx_of_first_one));
//
//            to_append_to.push_back(
//                    PartialFunction(
//                            function_size,
//                            new_template_for_output__function,
//                            template_for_output__partition
//                    )
//            );
//
//            template_for_output__function.set(idx_of_first_one, get_bit(other.total_function, idx_of_first_one));
//
//            other_contains_but_this_doesnt.set(idx_of_first_one, 0);
//        }
//    }
//    else
//    {
//        to_append_to.push_back(PartialFunction(function_size, total_function, partition));
//    }
//}

//void PartialFunction::append_intersection_with(PartialFunction other, vector<PartialFunction> &to_append_to) {
//    if(!has_empty_intersection_with(other))
//    {
//        to_append_to.push_back(
//                PartialFunction(
//                        function_size,
//                        (total_function & partition) | (other.total_function & other.partition),
//                        partition | other.partition
//                )
//        );
//    }
//}

bool PartialFunction::full()
{
    return partition.test_range(0, function_size - 1);
}


bool PartialFunction::operator < (const PartialFunction& other) const {
    if(total_function == other.total_function)
    {
        if(partition == other.partition)
        {
            return function_size < other.function_size;
        }
        return partition < other.partition;
    }
    return total_function < other.total_function;
}

bool PartialFunction::operator == (const PartialFunction& other) const {
    if(total_function == other.total_function)
    {
        if(partition == other.partition)
        {
            return function_size == other.function_size;
        }
        return false;
    }
    return false;
}
