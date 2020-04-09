//
// Created by Kliment Serafimov on 3/7/20.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_BITVECTOR_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_BITVECTOR_H


//
// Created by Kliment Serafimov on 1/19/20.
//

#include <bitset>

using namespace std;

#define BITVECTOR_SIZE 64

enum BitvectorConstructorType {all_zeroes, all_ones};

class Bitvector: public bitset<BITVECTOR_SIZE>
{
    bool size_defined = false;
    int size;
public:
    Bitvector() = default;
    Bitvector(BitvectorConstructorType type, int _size);
    Bitvector(bitset<BITVECTOR_SIZE> num);
    Bitvector(long long num);

    bool operator < (const int &other) const
    {
        return to_ullong() < other;
    }

    int get_size()
    {
        assert(size_defined);
        return size;
    }

    bool operator < (const Bitvector &other) const
    {
        for (int i = BITVECTOR_SIZE-1; i >= 0; i--) {
            if (test(i) ^ other.test(i)) return other.test(i);
        }
        return false;
    }

    void operator ++(int)
    {
        for(int i = 0;i < BITVECTOR_SIZE; i++)
        {
            if(!test(i))
            {
                set(i, 1);
                break;
            }
            else
            {
                set(i, 0);
            }
        }
//        unsigned long long ullong = to_ullong();
//        ullong++;
//        Bitvector ret = ullong;
//        return ret;
    }

    Bitvector get_flipped()
    {
        Bitvector ret = *this;
        ret.flip();
        return ret;
    }

    unsigned int num_trailing_zeroes();

    bool test_range(int begin, int end)
    {
        for(int i = begin; i<=end;i++)
        {
            if(!test(i))
            {
                return false;
            }
        }
        return true;
    }

    void set_range(int begin, int end)
    {
        if(size_defined)
        {
            size = max(size, end+1);
        }
        else
        {
            size = end+1;
        }
        for(int i = begin; i<=end;i++)
        {
            set(i);
        }
    }

    void set(int idx)
    {
        if(size_defined)
        {
            size = max(size, idx+1);
        }
        else
        {
            size = idx+1;
        }

        bitset<BITVECTOR_SIZE>::set(idx);
    }

    void set(int idx, int val)
    {
        if(size_defined)
        {
            size = max(size, idx+1);
        }
        else
        {
            size = idx+1;
        }
        bitset<BITVECTOR_SIZE>::set(idx, val);
    }


    void set_size(int i);

    string to_string()
    {
        string to_remove_0s = bitset<BITVECTOR_SIZE>::to_string();
        string ret;
        for(int i = to_remove_0s.size()-1; i >= to_remove_0s.size()-get_size() ;i--)
        {
            ret += to_remove_0s[i];
        }
        reverse(ret.begin(), ret.end());
        return ret;
    }
};

int get_bit(Bitvector bitvector, int idx);

string bitvector_to_str(int bitvector, int n);

string bitvector_to_str(Bitvector bitvector, int n);

#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_BITVECTOR_H
