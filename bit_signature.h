//
// Created by Kliment Serafimov on 2019-02-16.
//

#include "Header.h"

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_BIT_SIGNATURE_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_BIT_SIGNATURE_H


class bit_signature
{

    bool sort_by_defined = false;
    double sort_by = 0;

public:

    double value = 0;
    int num_operators = 0;
    //limit to 4 input bits;
    int gate = 0;
    vector<int> operands;

    int vector_id = 0;

    bool is_temporary = false;

    string print()
    {
        string ret = to_string(vector_id);
        if(num_operators == 2)
        {
            ret+= " "+bitset<4>(gate).to_string() + " ("+to_string(operands[0]) +", "+to_string(operands[1]) +")";
        }
        else
        {

        }
        return ret;
    }

    bit_signature(){}

    /*bit_signature(const bit_signature &to_copy)
    {
        sort_by_defined = to_copy.sort_by_defined;
        sort_by = to_copy.sort_by;
        value =to_copy.value;
        num_operators =to_copy.num_operators;
        gate =to_copy.gate;
        operands = to_copy.operands;
        vector_id = to_copy.vector_id;
    }*/

    bit_signature(double d)
    {
        value = d;
    }

    bit_signature(int id)
    {
        num_operators = 1;
        gate = IS_GATE;
        vector_id = id;
        operands.push_back(id);
    }

    bit_signature(double d, int id)
    {
        value = d;

        num_operators = 1;
        gate = IS_GATE;
        vector_id = id;
        operands.push_back(id);
    }

    bit_signature(double d, int first, int second)
    {
        value = d;
        num_operators = 2;
        operands.push_back(first);
        operands.push_back(second);
    }

    bit_signature(double d, int _gate, int first, int second)
    {
        value = d;
        num_operators = 2;
        gate = _gate;
        operands.push_back(first);
        operands.push_back(second);
    }
    bit_signature(double d, int _gate, vector<int> _operands)
    {
        value = d;
        num_operators = (int)_operands.size();
        gate = _gate;
        operands = _operands;
    }

    void flip_value()
    {
        assert(value == 1||value == 0);
        value = 1-value;
    }

    pair<int, int> operands_to_pair()
    {
        assert(operands.size() == 2);
        return make_pair(operands[0], operands[1]);
    }

    int to_01(vector<bit_signature> the_vec)
    {
        int ret = 0;
        for(int i = 0;i<operands.size();i++)
        {
            if(the_vec[operands[i]] == 1)
            {
                ret|=(1<<i);
            }
            else if(the_vec[operands[i]] == -1)
            {

            }
            else
            {
                assert(0);
            }
        }
        return ret;
    }

    void set_temporary()
    {
        assert(!is_temporary);
        is_temporary = true;
    }
    void unset_temprory()
    {
        assert(is_temporary);
        is_temporary = false;
    }

    void set_sort_by(double _new_sort_by)
    {
        sort_by_defined = true;
        sort_by = _new_sort_by;
    }

    double get_sort_by() const
    {
        return sort_by_defined*sort_by+(!sort_by_defined)*value;
    }

    bool operator < (const bit_signature &other) const
    {
        return get_sort_by() < other.get_sort_by();
    }

    operator double() const
    {
        return value;
    }

    void operator *= (double other)
    {
        value*=other;
    }
    void operator /= (double other)
    {
        value/=other;
    }
    void operator += (double other)
    {
        value+=other;
    }
    void operator -= (double other)
    {
        value+=other;
    }
    bool is(int other_gate)
    {
        return other_gate == gate;
    }
};

class operator_signature: public bit_signature
{
public:
    operator_signature(){}
    operator_signature(bit_signature x): bit_signature(x){}
    operator_signature(double val, int _gate, int first_operand, int second_operand): bit_signature(val)
    {
        gate =_gate;

        num_operators = 2;
        operands.push_back(first_operand);
        operands.push_back(second_operand);
    }
    operator_signature(int _gate, int first_operand, int second_operand)
    {
        gate =_gate;

        num_operators = 2;
        operands.push_back(first_operand);
        operands.push_back(second_operand);
    }
    operator_signature(int select_first, int select_second, int first_operand, int second_operand)
    {
        int selected_input = 0;
        selected_input|=(select_first<<0);
        selected_input|=(select_second<<1);

        gate = (1<<selected_input);

        num_operators = 2;
        operands.push_back(first_operand);
        operands.push_back(second_operand);
    }

    operator_signature(void (operator_signature::*funct)(), int _gate, int left, int right)
    {
        (this->*funct)();

        gate =_gate;

        num_operators = 2;
        operands.push_back(left);
        operands.push_back(right);
    }


    bit_signature apply_operator(vector<bit_signature> on_vector)
    {
        int input = to_01(on_vector);
        double val = 0;
        if(((1<<input)&gate) != 0)
        {
            val = 1;
        }
        else
        {
            val = -1;
        }
        bit_signature ret = *this;
        ret.value = val;
        return ret;
    }
};


#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_BIT_SIGNATURE_H
