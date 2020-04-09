//
// Created by Kliment Serafimov on 3/7/20.
//

#include "Bitvector.h"

int get_bit(Bitvector bitvector, int idx) {
    return bitvector.test(idx);
}

string bitvector_to_str(int bitvector, int n)
{
    string ret = "";
    for(int i = n-1;i>=0;i--)
    {
        if(get_bit(bitvector, i))
        {
            ret+="1";
        }
        else
        {
            ret+="0";
        }
    }
    return ret;
}

string bitvector_to_str(Bitvector bitvector, int n)
{
    assert(bitvector.get_size() == n);
    return bitvector.to_string();
}


Bitvector::Bitvector(BitvectorConstructorType type, int _size)
{
    size_defined = true;
    size = _size;
    if(type == all_ones)
    {
        for(int i = 0;i<size;i++)
        {
            set(i);
        }
    }
}

Bitvector::Bitvector(bitset<BITVECTOR_SIZE> _bitset): bitset<BITVECTOR_SIZE>(_bitset)
{
}

Bitvector::Bitvector(long long num): bitset<BITVECTOR_SIZE>(num)
{
    assert(num >= 0);
}

unsigned int Bitvector::num_trailing_zeroes() {
    int ret = 0;
    for(int i = 0;i<BITVECTOR_SIZE; i++)
    {
        if(test(i) == false)
        {
            ret++;
        } else {
            break;
        }
    }
    return ret;
}

void Bitvector::set_size(int _size)
{
    assert(!size_defined);
    for(int i = _size; i<BITVECTOR_SIZE; i++)
    {
        assert(!test(i));
    }
    size_defined = true;
    size = _size;
}
