//
// Created by Kliment Serafimov on 10/8/18.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DATA_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DATA_H

//
//  Data.hpp
//  neural guided decision tree
//
//  Created by Kliment Serafimov on 6/3/18.
//  Copyright © 2018 Kliment Serafimov. All rights reserved.
//

#include <stdio.h>
#include "Header.h"
#include "bit_signature.h"

class Data
{
public:

    /*struct and_operator: public operator_signature
    {
        and_operator(double val, int left, int right): operator_signature(val, AND_GATE, left, right){};
        and_operator(int left, int right): operator_signature(AND_GATE, left, right){};
        and_operator(void (and_operator::*funct)(), int left, int right): operator_signature(AND_GATE, left, right)
        {
            (this->*funct)();
        }
    };
    struct or_operator: public operator_signature
    {
        or_operator(double val, int left, int right): operator_signature(val, OR_GATE, left, right){};
        or_operator(int left, int right): operator_signature(OR_GATE, left, right){};
        or_operator(void (or_operator::*funct)(), int left, int right): operator_signature(OR_GATE, left, right)
        {
            (this->*funct)();
        }
    };
    struct nand_operator:public operator_signature
    {
        nand_operator(double val, int left, int right): operator_signature(val, NAND_GATE, left, right){};
        nand_operator(int left, int right): operator_signature(NAND_GATE, left, right){};
        nand_operator(void (nand_operator::*funct)(), int left, int right): operator_signature(NAND_GATE, left, right)
        {
            (this->*funct)();
        }
    };
    struct nor_operator:public operator_signature
    {
        nor_operator(double val, int left, int right): operator_signature(val, NOR_GATE, left, right){};
        nor_operator(int left, int right): operator_signature(NOR_GATE, left, right){};
        nor_operator(void (nor_operator::*funct)(), int left, int right): operator_signature(NOR_GATE, left, right)
        {
            (this->*funct)();
        }
    };
    */
    int sampleSize = 0;

    int numInputs = 0;
    int numOutputs = 0;

    vector<vector<bit_signature> > in;
    vector<vector<bit_signature> > out;

    vector<vector<bit_signature> > hidden_in;

    vector<operator_signature> circuit;

    typedef pair<bit_signature, int> unit_gate_type;

    vector<unit_gate_type> in_AND_constraint;
    vector<unit_gate_type> in_OR_constraint;

    void clear()
    {
        sampleSize = 0;
        numInputs = 0;
        numOutputs = 0;
        in.clear();
        out.clear();
        circuit.clear();
        in_AND_constraint.clear();
        in_OR_constraint.clear();
    }

    void printData(string s)
    {
        cout << s <<endl;
        for(int i = 0; i<size();i++)
        {
            cout << printInput(i) <<" "<< printOutput(i) <<endl;
        }
        cout << endl;
    }

    virtual string print()
    {
        return printConcatinateOutput();
    }




    string printConcatinateOutput()
    {
        string ret;
        for(int i = 0;i<out.size();i++)
        {
            ret+=printOutput(i);
        }
        return ret;
    }

    class normalizer
    {
    public:
        double scale_by = 1;
        double offset = 0;
    };

    normalizer the_normalizer;
    void normalize(Data& ret)
    {
        double min_dimension_value = 1000;
        double max_dimension_value = -1000;
        for(int i = 0;i<size();i++)
        {
            for(int j = 0;j<out[i].size();j++)
            {
                min_dimension_value = min(min_dimension_value, out[i][j].value);
                max_dimension_value = max(max_dimension_value, out[i][j].value);
            }
        }
        the_normalizer.offset = 0.25-min_dimension_value;
        the_normalizer.scale_by = 0.5/(max_dimension_value-min_dimension_value);

        ret = *this;

        ret.printData("Before normalization");
        for(int i = 0;i<size();i++)
        {
            for(int j = 0;j<out[i].size();j++)
            {
                ret.out[i][j].value+=the_normalizer.offset;
                ret.out[i][j].value*=the_normalizer.scale_by;
            }
        }
        ret.printData("After normalization");
    }

    void unnormalize(vector<bit_signature> &the_vector)
    {
        for(int i = 0;i<the_vector.size();i++)
        {
            the_vector[i].value*=1/the_normalizer.scale_by;
            the_vector[i].value-=the_normalizer.offset;
        }
    }

    int get_score(bool do_print)
    {
        int distance = 0;
        vector<int> dimensions_used;
        for(int i = 0; i<numOutputs;i++)
        {
            int min_local_distance = size();
            int dimension_used = -1;
            for(int j = 0;j<numInputs;j++)
            {
                int local_distance = 0;
                int local_injection_score = 0;
                for(int k = 0;k<size();k++)
                {
                    local_distance += (in[k][j] != out[k][i]);
                }
                if(min_local_distance>local_distance)
                {
                    dimension_used = j;
                    min_local_distance = min(min_local_distance, local_distance);
                }
            }
            dimensions_used.push_back(dimension_used);
            distance+=min_local_distance;
        }

        if(do_print)
        {

            for(int k = 0;k<size();k++)
            {
                cout << printInput(k) <<" " << printOutput(k) <<" ";
                for(int i = 0;i<numOutputs;i++)
                {
                    cout << (in[k][dimensions_used[i]] == out[k][i]);
                }
                cout << endl;
            }
        }
        return distance;
    }

    bool apply_new_operator_to_data(bit_signature& the_operator, Data& ret)
    {
        operator_signature use = the_operator;
        return apply_new_operator_to_data_operator(use, ret);
    }

    bool apply_new_operator_to_data(operator_signature& the_operator, Data& ret)
    {
        return apply_new_operator_to_data_operator(the_operator, ret);
    }

    bool apply_new_operator_to_data_operator(operator_signature& the_operator, Data& ret)
    {
        assert(ret.size() == 0);
        the_operator.vector_id = numInputs;
        for(int i = 0;i<in.size();i++)
        {
            vector<bit_signature> new_in = in[i];
            bit_signature new_bit = the_operator.apply_operator(in[i]);
            new_in.push_back(new_bit);
            ret.push_back(new_in, out[i]);
        }

        return !ret.is_redundant(numInputs);
    }

    void apply_new_operators_to_data(vector<operator_signature> the_operators, Data& ret)
    {
        assert(ret.size() == 0);

        for(int i = 0;i<in.size();i++)
        {
            ret.push_back(in[i], out[i]);
        }

        ret.apply_new_operators_to_data(the_operators);
    }

    void apply_new_operators_to_data(vector<operator_signature> the_operators)
    {
        for(int operator_id = 0;operator_id< the_operators.size(); operator_id++)
        {
            assert(the_operators[operator_id].is_temporary);
            assert(the_operators[operator_id].operands.size() == 2);
            assert(the_operators[operator_id].num_operators == 2);

            the_operators[operator_id].vector_id = circuit.size();


            for(int i = 0;i<in.size();i++)
            {
                operator_signature tmp = the_operators[operator_id].apply_operator(in[i]);
                assert(tmp.operands.size() == 2);
                assert(tmp.num_operators == 2);
                in[i].push_back(tmp);
            }

            numInputs++;


            if(is_redundant(numInputs-1))
            {

                the_operators[operator_id].vector_id = -1;
                for(int i = 0;i<in.size();i++)
                {
                    vector<bit_signature> replace_in;
                    for(int j = 0;j<in[i].size()-1;j++)
                    {
                        replace_in.push_back(in[i][j]);
                    }
                    in[i] = replace_in;
                }
                numInputs--;
            }
            else
            {

                circuit.push_back(the_operators[operator_id]);
            }
        }

    }

    bit_signature select_most_confusing_dimension()
    {
        vector<bit_signature> score;
        for(int i = 0;i<numInputs;i++)
        {
            for(int j = 0;j<size();j++)
            {

            }
        }
        return bit_signature();
    }

    bool is_redundant(int bit_id)
    {
        vector<int> count_same(numInputs, 0);
        vector<bool> has_0(numInputs, 0);
        vector<bool> has_1(numInputs, 0);

        for(int i = 0;i<in.size();i++)
        {
            for(int j = 0;j<in[i].size();j++)
            {
                count_same[j] += (in[i][j] == in[i][bit_id]);
            }

            has_0[bit_id] = has_0[bit_id]|(in[i][bit_id] == -1);
            has_1[bit_id] = has_1[bit_id]|(in[i][bit_id] == 1);
        }

        bool ret = false;

        for(int i = 0;i<numInputs;i++)
        {
            if(count_same[i] == size() && i != bit_id)
            {
                ret = true;
                break;
            }

            if(count_same[i] == 0 && i!= bit_id)
            {
                ret = true;
                break;
            }

        }

        if(has_0[bit_id] == 0 || has_1[bit_id] == 0)
        {
            ret = true;
        }

        return ret;
    }

    bit_signature add_single_kernel_to_base_and_discard_rest(bit_signature the_bit, Data& ret)
    {
        assert(circuit.size() == numInputs);
        vector<bit_signature> new_circuit;
        bit_signature ret_bit = the_bit;
        bool enter_circuit = false;
        for(int j = 0;j<circuit.size();j++)
        {
            if(j == the_bit.vector_id)
            {
                enter_circuit = true;
                int local_bit = (int)new_circuit.size();
                operator_signature new_gate = circuit[j];
                new_gate.vector_id = local_bit;
                if(new_gate.is_temporary) new_gate.unset_temprory();
                new_circuit.push_back(new_gate);
                ret_bit = new_gate;
            }
            else if(!circuit[j].is_temporary)//in[i][j].is(IS_GATE))
            {
                int local_bit = (int)new_circuit.size();
                operator_signature new_gate = circuit[j];
                new_gate.vector_id = local_bit;
                new_circuit.push_back(new_gate);
            }

        }

        for(int i = 0;i<in.size();i++)
        {
            vector<bit_signature> new_in;
            for(int j = 0;j<in[i].size();j++)
            {
                if(j == the_bit.vector_id)
                {
                    int local_bit = (int)new_in.size();
                    operator_signature new_gate = in[i][j];
                    new_gate.vector_id = local_bit;
                    if(new_gate.is_temporary)new_gate.unset_temprory();
                    new_in.push_back(new_gate);
                }
                else if(!in[i][j].is_temporary)//in[i][j].is(IS_GATE))
                {
                    int local_bit = (int)new_in.size();
                    operator_signature new_gate = in[i][j];
                    new_gate.vector_id = local_bit;
                    new_in.push_back(new_gate);
                }
            }
            ret.push_back(new_in, out[i]);
        }
        numInputs = (int)circuit.size();
        return ret_bit;
    }

    void and_or_exstention(Data &ret, vector<bit_signature> bits_wanted, int top_bits)
    {
        vector<int> old_active_bits = get_active_bits();

        vector<int> active_bits;

        for(int i = 0;i<min(top_bits, (int)bits_wanted.size());i++)
        {
            bool enter = false;
            for(int j = 0;j<old_active_bits.size();j++)
            {
                if(old_active_bits[j] == bits_wanted[i].vector_id)
                {
                    enter = true;
                    break;
                }
            }
            if(enter)
            {
                active_bits.push_back(bits_wanted[i].vector_id);
            }
            else
            {
                assert(bits_wanted[i].get_sort_by() == 0 || isnan(bits_wanted[i].get_sort_by()));
            }
        }

        apply_new_operators_to_data(get_operators(active_bits), ret);

    }

    void apply_important_pair_dimensions_to_data(vector<pair<int, int> > operands, Data& ret)
    {
        apply_new_operators_to_data(get_operators(operands), ret);
    }

    void apply_dnf_important_pair_dimensions_to_data(vector<pair<int, int> > operands, Data& ret)
    {
        /*vector<operator_signature> all_new_operators;
        for(int id = 0;id<operands.size();id++)
        {
            int or_gate = 14;

            //int gates[5][10] = {{or_gate, -1}, {1, 2, 4, or_gate, 8, -1}, {2, or_gate, 8, -1}, {4, or_gate, 8, -1}, {or_gate, 8, -1}};

            int and_gate = 8;
            int gates[5][10] = {{and_gate, -1}, {15-1, 15-2, 15-4, and_gate, 15-8, -1}, {15-2, and_gate, 15-8, -1}, {15-4, and_gate, 15-8, -1}, {and_gate, 15-8, -1}};


            int f_dim = operands[id].f;
            int s_dim = operands[id].s;

            vector<int> gate_buckets;
            int type;

            assert(circuit[f_dim].num_operators >= 1 && circuit[s_dim].num_operators >= 1);

            if(circuit[f_dim].gate == or_gate || circuit[s_dim].gate == or_gate)
            {
                assert(circuit[f_dim].num_operators == 2 || circuit[s_dim].num_operators == 2);

                type = 0;
            }
            else if(circuit[f_dim].num_operators == 1 && circuit[s_dim].num_operators == 1)
            {
                assert(circuit[f_dim].gate != or_gate && circuit[s_dim].gate != or_gate);
                 type = 1;
            }
            else if(circuit[f_dim].num_operators == 1)
            {
                assert(circuit[s_dim].num_operators == 2);
                assert(circuit[f_dim].gate != or_gate && circuit[s_dim].gate != or_gate);
                 type = 2;
            }
            else if(circuit[s_dim].num_operators == 1)
            {
                assert(circuit[f_dim].gate != or_gate && circuit[s_dim].gate != or_gate);
                assert(circuit[f_dim].num_operators == 2);
                type = 3;
            }
            else
            {
                assert(circuit[f_dim].num_operators == 2 && circuit[s_dim].num_operators == 2);
                assert(circuit[f_dim].gate != or_gate && circuit[s_dim].gate != or_gate);
                type = 4;
            }
            for(int i = 0;gates[type][i]!=-1; i++)
            {
                all_new_operators.push_back(operator_signature(&bit_signature::set_temporary, gates[type][i], f_dim, s_dim));
            }
        }*/

        apply_new_operators_to_data(get_operators(operands), ret);
    }

    vector<operator_signature> get_operators(vector<pair<int, int> > operands)
    {
        vector<operator_signature> all_new_operators;
        for(int id = 0;id<operands.size();id++)
        {
            //int gates[10] = {1, 7, 8, 14}; int num_gates = 4;
            int gates[10] = {1, 2, 4, 6, 7, 8, 9, 11, 13, 14}; int num_gates = 10;
            for(int i = 0;i<num_gates;i++)
            {
                all_new_operators.push_back(operator_signature(&bit_signature::set_temporary, gates[i], operands[id].f, operands[id].s));
            }
        }
        return all_new_operators;
    }


    vector<operator_signature> get_operators(vector<int> active_bits)
    {

        vector<operator_signature> all_new_operators;
        for(int j = 0;j<active_bits.size();j++)
        {
            for(int k = j+1; k<active_bits.size();k++)
            {
                //int gates[10] = {1, 2, 4, 6, 7, 8, 9, 11, 13, 14};
                int gates[10] = {1, 7, 8, 14};
                //int gates[10] = {6, 9};
                for(int i = 0;i<4;i++)
                {
                    all_new_operators.push_back(operator_signature(&bit_signature::set_temporary, gates[i], active_bits[j], active_bits[k] ));
                }
                /*
                all_new_operators.push_back(and_operator(&bit_signature::set_temporary, active_bits[j], active_bits[k]));

                all_new_operators.push_back(or_operator(&bit_signature::set_temporary, active_bits[j], active_bits[k]));

                all_new_operators.push_back(nand_operator(&bit_signature::set_temporary, active_bits[j], active_bits[k]));

                all_new_operators.push_back(nor_operator(&bit_signature::set_temporary, active_bits[j], active_bits[k]));

                all_new_operators.push_back(xor_operator(&bit_signature::set_temporary, active_bits[j], active_bits[k]));

                all_new_operators.push_back(xnot_operator(&bit_signature::set_temporary, active_bits[j], active_bits[k]));*/


            }
        }
        return all_new_operators;
    }

    void add_gates_between_active_bits(vector<int> active_bits, Data &ret)
    {
        //cout << "acttive bits.size = " << active_bits.size() << endl;


        /*for(int i = 0;i<in.size();i++)
        {
            vector<bit_signature> signature;
            for(int j = 0;j<in[i].size();j++)
            {
                signature.push_back(bit_signature(in[i][j], j));
            }


            for(int j = 0;j<active_bits.size();j++)
            {
                for(int k = j+1; k<active_bits.size();k++)
                {
                    double val;


                    if(in[i][active_bits[j]] == 1 && in[i][active_bits[k]] == 1)
                    {
                        val = 1;
                    }
                    else
                    {
                        val = -1;
                    }
                    signature.push_back(and_signature(val, active_bits[j], active_bits[k]));

                    if(in[i][active_bits[j]] == 1 || in[i][active_bits[k]] == 1)
                    {
                        val = 1;
                    }
                    else
                    {
                        val = -1;
                    }
                    signature.push_back(or_signature(val, active_bits[j], active_bits[k]));


                    if(in[i][active_bits[j]] == -1 || in[i][active_bits[k]] == -1)
                    {
                        val = 1;
                    }
                    else
                    {
                        val = -1;
                    }
                    signature.push_back(nand_signature(val, active_bits[j], active_bits[k]));


                    if(in[i][active_bits[j]] == -1 && in[i][active_bits[k]] == -1)
                    {
                        val = 1;
                    }
                    else
                    {
                        val = -1;
                    }
                    signature.push_back(nor_signature(val, active_bits[j], active_bits[k]));

                }
            }
            if(false)
            {
                int last_prev_id = (int)signature.size();
                for(int j = (int)in[i].size();j<last_prev_id;j++)
                {
                    for(int k = j+1;k<last_prev_id;k++)
                    {
                        double val;
                        if(signature[j] == 1 || signature[k] == 1)
                        {
                            val = 1;
                        }
                        else
                        {
                            val = -1;
                        }
                        signature.push_back(or_signature(val, j, k));
                    }
                }
            }
            ret.push_back(signature, out[i]);
            //cout << new_in.size() <<" ";
            //cout << ret.printInput(i) <<" "<< ret.printOutput(i) <<endl;
        }*/
    }

    vector<int> get_active_bits()
    {
        vector<int> ret;
        for(int j = 0;j<numInputs;j++)
        {
            for(int i = 1;i<in.size();i++)
            {
                if(in[i][j]!=in[i-1][j])
                {
                    ret.push_back(j);
                    break;
                }
            }
        }
        sort_v(ret);
        return ret;
    }

    int num_active_input_bits()
    {
        int ret = 0;
        for(int j = 0;j<numInputs;j++)
        {
            for(int i = 1;i<in.size();i++)
            {
                if(in[i][j]!=in[i-1][j])
                {
                    ret++;
                    break;
                }
            }
        }
        return ret;
    }

    pair<bit_signature, bit_signature> get_first_active_input_pair()
    {
        bit_signature first_bit = bit_signature(-1);
        for(int j = 0;j<numInputs;j++)
        {
            for(int i = 1;i<in.size();i++)
            {
                if(in[i][j]!=in[i-1][j])
                {
                    if(first_bit.vector_id == -1)
                    {
                        first_bit = bit_signature(j);
                    }
                    else
                    {
                        return make_pair(first_bit, bit_signature(j));
                    }
                }
            }
        }
        return make_pair(first_bit, first_bit);
    }

    bit_signature get_first_active_input_bit()
    {
        for(int j = 0;j<numInputs;j++)
        {
            for(int i = 1;i<in.size();i++)
            {
                if(in[i][j].value != in[i-1][j].value)
                {
                    bit_signature ret;
                    ret.vector_id = j;
                    return ret;
                }
            }
        }
        return -1;
    }

    bool is_constant()
    {
        for(int i = 1; i<out.size();i++)
        {
            for(int j = 0;j<out[i].size();j++)
            {
                if(out[i][j]!=out[i-1][j])
                {
                    return false;
                }
            }
        }
        return true;
    }

    void make_neutral(Data& ret)
    {
        for(int i = 0;i<size();i++)
        {
            vector<bit_signature> new_out(out[i].size(), 0);
            ret.push_back(in[i], new_out);
        }
    }

    int size()
    {
        return sampleSize;
    }
    void push_back(vector<bit_signature> new_in, vector<bit_signature> new_out)
    {
        assert(new_in.size() >= 0 && new_out.size() > 0);
        if(numInputs == 0 && new_in.size() >= 0 && numOutputs == 0 && new_out.size() != 0)
        {
            numInputs = (int)new_in.size();
            numOutputs = (int)new_out.size();
            for(int i = 0;i<numInputs; i++)
            {
                circuit.push_back(new_in[i]);
                circuit[i].value = 0;
                assert(circuit[i].vector_id == i);
            }
        }
        assert(numInputs == new_in.size());
        if(numOutputs != new_out.size())
        {
            cout << numInputs << " "<< new_in.size() <<endl;
            cout << numOutputs << " "<< new_out.size() <<endl;
        }
        assert(numOutputs == new_out.size());
        in.push_back(new_in);
        out.push_back(new_out);
        hidden_in.push_back(vector<bit_signature>());
        sampleSize++;
        assert(sampleSize == in.size());
    }

/*    Data(int _numInputs, int _numOutputs, vector<pair<bit_signature, int> > _constraint, string constraint_type)
    {
        numInputs = _numInputs;
        numOutputs = _numOutputs;
        if(constraint_type == "AND")
        {
            in_AND_constraint = _constraint;
        }
        if(constraint_type == "OR")
        {
            in_OR_constraint = _constraint;
        }
    }
 */

    void split(bit_signature split_idx, Data& first, Data& second)
    {
        vector<unit_gate_type> split_idx_vec;
        split_idx_vec.push_back(make_pair(split_idx, 1));
        split(split_idx_vec, first, second);
    }

    void split(operator_signature split_idx, Data& first, Data& second)
    {
        vector<unit_gate_type> split_idx_vec;
        split_idx_vec.push_back(make_pair(split_idx, 1));
        split(split_idx_vec, first, second);
    }


    void split(vector<unit_gate_type> split_idx, Data& first, Data& second)
    {
        //first = Data(numInputs, numOutputs, split_idx, "AND");
        //second = Data(numInputs, numOutputs, split_idx, "OR");

        for(int i = 0;i<in.size();i++)
        {
            assert(i<out.size());
            int num_matched = 0;
            int num_unmatched = 0;
            for(int j = 0;j<split_idx.size();j++)
            {
                assert(in[i].size() > split_idx[j].f.vector_id);
                int bit = in[i][split_idx[j].f.vector_id];
                if(bit == split_idx[j].s)
                {
                    num_matched++;
                }
                else
                {
                    num_unmatched++;
                }
            }
            if(num_matched == split_idx.size())//and
            {
                first.push_back(in[i], out[i]);
            }
            else
            {
                second.push_back(in[i], out[i]);
            }
        }
    }
    void split_and_remove_bit(bit_signature split_id, Data& first, Data& second)
    {
        split_and_remove_bit(make_pair(split_id, 1), first, second);
    }

    void split_and_remove_bit(unit_gate_type split_id, Data& first, Data& second)
    {
        assert(in.size()==out.size());
        for(int i = 0;i<in.size();i++)
        {
            assert(in[i].size() > split_id.f.vector_id);

            vector<bit_signature> new_in;
            for(int j = 0, new_j = 0;j<in[i].size();j++)
            {
                if(split_id.f.vector_id == j) {
                    //continue since we remove this bit
                }
                else {
                    new_in.pb(bit_signature(in[i][j].value, new_j));
                    new_j++;
                }
            }

            int bit = in[i][split_id.f.vector_id];
            if(bit == split_id.s)
            {
                first.push_back(new_in, out[i]);
            }
            else
            {
                second.push_back(new_in, out[i]);
            }
        }
    }

    string type;

    void make_binary()
    {
        for(int i=0;i<in.size();i++)
        {
            for(int j=0;j<in[i].size();j++)
            {
                if(in[i][j] == -1)
                {
                    in[i][j] = 0;
                }
            }
        }
    }

    Data()
    {

    }

    virtual void init_exaustive_table_with_unary_output(int num_bits, long long output)
    {
        for(int i = 0;i<(1<<num_bits);i++)
        {
            vector<bit_signature> new_in;
            for(int j = 0;j<num_bits; j++)
            {
                int val = ((i&(1ll<<j)) != 0);
                if(val == true)
                {
                    new_in.push_back(bit_signature(1.0, j));
                }
                else
                {
                    new_in.push_back(bit_signature(-1.0, j));
                }
            }
            vector<bit_signature> new_out;
            double out_bit = (double) (((output&(1<<i)))!=0);
            //if(out_bit == 0) out_bit = -1;
            new_out.push_back(bit_signature(out_bit, 0));

            push_back(new_in, new_out);
        }
    }

    double entropy_measure(int num_of_category_elements_in_bucket, int num_of_elements_in_bucket)
    {
        double ratio = 0;
        if(num_of_elements_in_bucket != 0)
            ratio = (double)num_of_category_elements_in_bucket/num_of_elements_in_bucket;
        double ret = 0;
        if(ratio != 0)
            ret = -ratio*log2(ratio);

        assert(0<=ret && ret <=1);

        return ret;
    }

    bit_signature get_most_entropic_input_bit()
    {
        pair<double, bit_signature> most_entropic_input_bit = mp(1000, bit_signature(-1));
        for(int i = 0;i<numInputs;i++)
        {
            map<vector<bit_signature>, pair<int, int> > category_sizes;
            pair<int, int>  total_size = mp(0, 0);

            for(int j = 0;j<size();j++)
            {
                if(category_sizes.find(out[j]) == category_sizes.end())
                {
                    category_sizes[out[j]] = mp(0, 0);
                }

                if(in[j][i] == 1.0)
                {
                    category_sizes[out[j]].f++;
                    total_size.f++;
                }
                else if(in[j][i] == -1.0)
                {
                    category_sizes[out[j]].s++;
                    total_size.s++;
                }
                else
                {
                    assert(0);
                }

            }

            if(total_size.f != 0 && total_size.s != 0)
            {

                pair<double, double> entropy;
                for(map<vector<bit_signature>, pair<int, int> >::iterator it = category_sizes.begin(); it != category_sizes.end(); it++)
                {
                    assert(2*entropy_measure(1, 2) == 1);
                    entropy.f+=entropy_measure((*it).s.f, total_size.f);
                    entropy.s+=entropy_measure((*it).s.s, total_size.s);
                    //cout << printVector((*it).f) << " :: " << (*it).s.f <<" "<< (*it).s.s << endl;
                }
                double score = (entropy.f+entropy.s);

                assert(0<=score && score <=2);
                //cout << score;
                //cout << endl;
                //cout << score <<" ";
                most_entropic_input_bit = min(most_entropic_input_bit, mp(score, bit_signature(i)));

            }

        }
        //cout << endl;

        //cout << endl;
        //cout << most_entropic_input_bit.s.vector_id <<endl;
        return most_entropic_input_bit.s;
    }

    void generateData(int &ret_numInputs, int &ret_numOutputs, string &ret_typ);

    void printTest(int id)
    {

        cout << "Hint number: " << id <<endl;
        cout << "in : ";
        for(int j=0; j<in[id].size(); j++)
        {
            cout << in[id][j]+(in[id][j]==-1)<<" ";
        }
        cout << endl<<"out: ";
        for(int i=0; i<out[id].size(); i++)
        {
            cout << out[id][i]<<" ";
        }
        cout << endl;
    }
    void printTest(int id, vector<bit_signature> prediction )
    {
        printTest(id);
        cout << "try: ";
        for(int j=0; j<prediction.size(); j++)
        {
            cout << (bool)(prediction[j]>0.5)<<" ";
        }
        cout << endl;
    }

    string printVector(vector<bit_signature> v)
    {
        string ret;
        bool is_binary = true;
        for(int i=0;i<v.size();i++)
        {
            if(v[i]!=1 && v[i]!=-1 && v[i]!=0)
            {
                is_binary = false;
                break;
            }
            ret+=((v[i]+(v[i]==-1))+'0');
        }
        if(!is_binary)
        {
            ret.clear();
            ret+="{ ";
            for(int i = 0;i<v.size();i++)
            {
                if(i!= 0)
                {
                    ret+=", ";
                }
                ret+=to_string(v[i].value);
            }
            ret+="}";
        }
        return ret;
    }
    string printInput(int id)
    {
        return printVector(in[id]);
    }
    string printOutput(int id)
    {
        return printVector(out[id]);
    }
    void seeData()
    {
        cout << "Sample size: " << sampleSize << endl;
        for(int i=0;i<sampleSize;i++)
        {
            cout << printInput(i) <<" "<< printOutput(i) <<endl;
        }
    }
    int count_wrong_bits(int id, vector<bit_signature> predict)
    {
        assert(out[id].size()==predict.size());
        int ret = 0;
        for(int i=0; i<predict.size(); i++)
        {
            if((predict[i]>0.5)!=(out[id][i]>0.5))
            {
                ret++;
            }
        }
        return ret;
    }
    bool checkPrediction(int id, vector<bit_signature> predict)
    {
        return count_wrong_bits(id, predict) == 0;
    }
};
class DecisionTreeScore
{
public:
    double size = -1;
    int num_solutions = -1;

    vector<string> dt_strings;
    vector<string> if_format_strings;

    DecisionTreeScore()
    {

    }

    DecisionTreeScore(double _size, string _decision_tree_in_string)
    {
        size = _size;
        dt_strings.pb(_decision_tree_in_string);
    }

    bool operator < (const DecisionTreeScore& other) const
    {
        /*if(size == other.size)
        {
            return num_solutions > other.num_solutions;
        }*/
        return size < other.size;
    }

    bool operator == (const DecisionTreeScore& other) const
    {
        return size == other.size && num_solutions == other.num_solutions;
    }

    string print()
    {
        return std::to_string((int)size);
        //return "("+std::to_string((int)size) + " " + std::to_string(num_solutions)+")";
    }

    string print(int id)
    {
        return "("+std::to_string((int)size) + + ", id=" + std::to_string(id)+")";
    }

};



class DataAndScore: public Data
{
public:

    DecisionTreeScore score;

    DataAndScore(DecisionTreeScore _score): Data()
    {
        score = _score;
    }

    DataAndScore(): Data()
    {

    }


    virtual string print() override
    {

        return  printConcatinateOutput() + " " + score.print();

    }

    bool operator < (const DataAndScore& other) const
    {
        return score < other.score;
    }

};

class DataAndNeuralError: public DataAndScore
{
public:
    double neuralError;

    DataAndNeuralError(): DataAndScore()
    {

    }
};


/*if(in[i][active_bits[j]] == 1 && in[i][active_bits[k]] == 1)
 {
 val = 1;
 }
 else
 {
 val = -1;
 }
 signature.push_back(and_signature(val, active_bits[j], active_bits[k]));*/



/*if(in[i][active_bits[j]] == 1 || in[i][active_bits[k]] == 1)
 {
 val = 1;
 }
 else
 {
 val = -1;
 }
 signature.push_back(or_signature(val, active_bits[j], active_bits[k]));*/


/*if(in[i][active_bits[j]] == -1 || in[i][active_bits[k]] == -1)
 {
 val = 1;
 }
 else
 {
 val = -1;
 }
 signature.push_back(nand_signature(val, active_bits[j], active_bits[k]));*/


/*if(in[i][active_bits[j]] == -1 && in[i][active_bits[k]] == -1)
 {
 val = 1;
 }
 else
 {
 val = -1;
 }
 signature.push_back(nor_signature(val, active_bits[j], active_bits[k]));*/


/*Data imporant_bit_expansion(int the_bit)
 {
 assert(numInputs >= 2);
 Data ret;
 //ret.numInputs = (numInputs-1)*2;
 for(int i = 0;i<in.size();i++)
 {
 vector<bit_signature> new_in;
 for(int j = 0;j<in[i].size();j++)
 {
 new_in.push_back(in[i][j]);
 }
 for(int j = 0;j<in[i].size();j++)
 {
 if(the_bit != j)
 {
 if(in[i][j] == 1 || in[i][the_bit] == 1.0)
 {
 new_in.push_back(1.0);
 }
 else
 {
 new_in.push_back(-1.0);
 }
 }
 }
 for(int j = 0;j<in[i].size();j++)
 {
 if(the_bit != j)
 {
 if(in[i][j] == 1 || in[i][the_bit] == -1)
 {
 new_in.push_back(1);
 }
 else
 {
 new_in.push_back(-1);
 }
 }
 }
 ret.push_back(new_in, out[i]);
 }
 for(int i = 0;i<ret.size();i++)
 {
 cout << ret.printInput(i) << " "<< ret.printOutput(i) << endl;
 }
 return ret;
 }*/


#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DATA_H
