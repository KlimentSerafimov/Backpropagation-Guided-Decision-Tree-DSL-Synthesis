//
//  Data.cpp
//  neural guided decision tree
//
//  Created by Kliment Serafimov on 6/3/18.
//  Copyright © 2018 Kliment Serafimov. All rights reserved.
//

#include "Data.h"


int getBit(int bit, int at)
{
    assert(0<=at&&at<=30);
    return ((bit&(1<<at))!=0);
}


void longestSubstring_dai_di_is_0
        (int &numInputs, int &numOutputs, int &sampleSize, string &type, vector<vector<bit_signature> > &in, vector<vector<bit_signature> > &out)
{
    type = "longestSubstring_dai_di_is_0";

    assert(1<=numInputs&&numInputs<=30);

    int universe = (1<<numInputs);
    sampleSize = 0;
    numOutputs = numInputs;

    for(int i=0; i<universe; i++, sampleSize++)
    {
        vector<bit_signature> sample;
        vector<bit_signature> sampleOut;

        for(int j=0; j<numInputs; j++)
        {
            int a = getBit(i, j);
            if(a==0)
            {
                a=-1;
            }
            sample.pb(a);
        }
        rev_v(sample);
        int ret = 1, pos = 0, now = 1;
        for(int j=1; j<numOutputs; j++)
        {
            if(sample[j-1]==sample[j])
            {
                now++;
                if(ret <= now)
                {
                    pos = j;
                    ret = now;
                }
            }
            else
            {
                now = 1;
            }
        }
        for(int j=0; j<numOutputs; j++)
        {
            if(pos-ret<j&&j<=ret)
            {
                sampleOut.pb(1);
            }
            else
            {
                sampleOut.pb(0);
            }
        }
        in.pb(sample);
        out.pb(sampleOut);
    }
}

void longestSubstring_ai_is_1
        (int &numInputs, int &numOutputs, int &sampleSize, string &type, vector<vector<bit_signature> > &in, vector<vector<bit_signature> > &out)
{
    type = "longestSubstring_ai_is_1";

    assert(1<=numInputs&&numInputs<=30);

    int universe = (1<<numInputs);
    sampleSize = 0;
    numOutputs = 0;

    numOutputs = numInputs+1;
    /*int toFindNumBits = numInputs;
     while(toFindNumBits>0)
     {
     toFindNumBits/=2;
     numOutputs++;
     }*/

    for(int latice=0; latice<universe; latice++, sampleSize++)
    {
        vector<bit_signature> sample;
        vector<bit_signature> sampleOut;

        for(int j=0; j<numInputs; j++)
        {
            int a = getBit(latice, j);
            if(a==0)
            {
                a=-1;
            }
            sample.pb(bit_signature(a, j));
        }
        //rev_v(sample);
        int ret = 0, pos = -1, now = 0;
        for(int j=0; j<numInputs; j++)
        {
            if(sample[j]==1)
            {
                now++;
                if(ret <= now)
                {
                    pos = j;
                    ret = now;
                }
            }
            else
            {
                now = 0;
            }
        }
        for(int j=0; j<numOutputs; j++)
        {
            //sampleOut.pb(getBit(ret, j));
            if(j<=ret)
            {
                sampleOut.pb(bit_signature(1, j));
            }
            else
            {
                sampleOut.pb(bit_signature(0, j));
            }
        }
        in.pb(sample);
        out.pb(sampleOut);
    }
}

void longest_substring_of_two_strings
        (int &numInputs, int &numOutputs, int &sampleSize, string &type, vector<vector<bit_signature> > &in, vector<vector<bit_signature> > &out)
{
    type = "longestSubstring_ai_is_1";

    assert(1<=numInputs&&numInputs<=30);

    int universe = (1<<numInputs);
    sampleSize = 0;
    numOutputs = 0;

    numOutputs = (numInputs+1)/2;
    /*int toFindNumBits = numInputs;
     while(toFindNumBits>0)
     {
     toFindNumBits/=2;
     numOutputs++;
     }*/

    for(int latice=0; latice<universe; latice++, sampleSize++)
    {
        vector<bit_signature> sample;
        vector<bit_signature> sampleOut;

        for(int j=0; j<numInputs; j++)
        {
            int a = getBit(latice, j);
            if(a==0)
            {
                a=-1;
            }
            sample.pb(a);
        }
        rev_v(sample);
        int ret = 0, pos = -1, now = 0;
        for(int j=0; j<numInputs; j++)
        {
            if(sample[j]==1)
            {
                now++;
                if(ret <= now)
                {
                    pos = j;
                    ret = now;
                }
            }
            else
            {
                now = 0;
            }
            if(j == (numInputs+1)/2-1)
            {
                if(ret <= now)
                {
                    pos = j;
                    ret = now;
                }
                now = 0;
            }
        }
        for(int j=0; j<numOutputs; j++)
        {
            //sampleOut.pb(getBit(ret, j));
            if(j<=ret)
            {
                sampleOut.pb(1);
            }
            else
            {
                sampleOut.pb(0);
            }
        }
        in.pb(sample);
        out.pb(sampleOut);
    }
}

void longest_substring_double_to_middle
        (int &numInputs, int &numOutputs, int &sampleSize, string &type, vector<vector<bit_signature> > &in, vector<vector<bit_signature> > &out)
{
    type = "longestSubstring_ai_is_1";

    assert(1<=numInputs&&numInputs<=30);

    int universe = (1<<numInputs);
    sampleSize = 0;
    numOutputs = 0;

    numOutputs = 1+numInputs+numInputs/2;
    /*int toFindNumBits = numInputs;
     while(toFindNumBits>0)
     {
     toFindNumBits/=2;
     numOutputs++;
     }*/

    for(int latice=0; latice<universe; latice++, sampleSize++)
    {
        vector<bit_signature> sample;
        vector<bit_signature> sampleOut;

        for(int j=0; j<numInputs; j++)
        {
            int a = getBit(latice, j);
            if(a==0)
            {
                a=-1;
            }
            sample.pb(a);
        }
        rev_v(sample);
        int ret = 0, pos = -1, now = 0;
        for(int j=0; j<numInputs; j++)
        {
            if(sample[j]==1)
            {
                now++;
                if(j == (numInputs+1)/2-1)
                {
                    now*=2;
                }
                if(ret <= now)
                {
                    pos = j;
                    ret = now;
                }
            }
            else
            {
                now = 0;
            }

        }
        for(int j=0; j<numOutputs; j++)
        {
            //sampleOut.pb(getBit(ret, j));
            if(j<=ret)
            {
                sampleOut.pb(1);
            }
            else
            {
                sampleOut.pb(0);
            }
        }
        in.pb(sample);
        out.pb(sampleOut);
    }
}

void a_i_is_a_i_plus_one_for_all
        (int &numInputs, int &numOutputs, int &sampleSize, string &type, vector<vector<bit_signature> > &in, vector<vector<bit_signature> > &out)
{
    type = "a_i_is_a_i_plus_one_for_all";
    assert(1<=numInputs&&numInputs<=30);

    int universe = (1<<numInputs);
    sampleSize = 0;
    numOutputs = numInputs-1;

    for(int i=0; i<universe; i++, sampleSize++)
    {
        vector<bit_signature> sample;
        vector<bit_signature> sampleOut;

        for(int j=0; j<numInputs; j++)
        {
            int a = getBit(i, j);
            if(a==0)
            {
                a=-1;
            }
            sample.pb(a);
        }
        rev_v(sample);
        for(int j=0; j<numOutputs; j++)
        {
            if(sample[j]==sample[j+1])
            {
                sampleOut.pb(1);
            }
            else
            {
                sampleOut.pb(0);
            }
        }
        in.pb(sample);
        out.pb(sampleOut);
    }
}

void nullFunction(int &numInputs, int &numOutputs, int &sampleSize, string &type, vector<vector<bit_signature> > &in, vector<vector<bit_signature> > &out){}

void input_is_output(int &numInputs, int &numOutputs, int &sampleSize, string &type, vector<vector<bit_signature> > &in, vector<vector<bit_signature> > &out)
{
    type = "input_is_output";
    sampleSize = (1<<numInputs);
    numOutputs = numInputs;
    for(int i=0;i<sampleSize;i++)
    {
        vector<bit_signature> sample;
        vector<bit_signature> sampleOut;
        for(int j=0;j<numInputs;j++)
        {
            double x = ((i&(1<<j))!=0);
            sampleOut.pb(x);
            x -= (x==0);
            sample.pb(x);
        }
        rev_v(sample);
        rev_v(sampleOut);
        in.pb(sample);
        out.pb(sampleOut);
    }
}

void vector_loop_language(int &numInputs, int &numOutputs, int &sampleSize, string &type, vector<vector<bit_signature> > &in, vector<vector<bit_signature> > &out)
{
    assert((int)sqrt(numInputs)*(int)sqrt(numInputs)== numInputs);
    numOutputs = numInputs;
    numInputs = 2*sqrt(numInputs);
    type = "vector_loop_language";
    int state[6] = {0, 0, 0, 0, 0, 0};
    int L[7][2][6]
            =
            {
                    {{0, 0, 1, 1, 0, 1}, {0, 1, 0, 0, 1, 0}},
                    {{0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 1}},
                    {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 0}},
                    {{0, 1, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0}},
                    {{0, 0, 0, 0, 1, 0}, {1, 0, 0, 0, 0, 0}},
                    {{0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0}}
            };

    for(int i = 0; i <= numInputs/2; i++)
    {
        for(int j = 0;j <= numInputs/2; j++)
        {
            memset(state, 0, sizeof(state));
            state[0] = i;
            state[1] = j;
            bool enter = true;
            while(enter)
            {
                enter = false;
                for(int k = 0;k<6 && !enter;k++)
                {
                    bool take = true;
                    for(int l = 0;l<6;l++)
                    {
                        if(L[k][1][l] > state[l])
                        {
                            take = false;
                        }
                    }
                    if(take)
                    {
                        for(int l = 0;l<6;l++)
                        {
                            state[l] -= L[k][1][l];
                            state[l] |= L[k][0][l];
                        }
                        enter = true;
                        cout << "take " << k <<endl;
                    }
                }
            }

            vector<bit_signature> local_in;
            vector<bit_signature> local_out;
            for(int k = 0;k<numInputs/2;k++)
            {
                if(k<i)
                    local_in.push_back(1);
                else
                {
                    local_in.push_back(-1);
                }
            }
            for(int k = 0;k<numInputs/2;k++)
            {
                if(k<j)
                    local_in.push_back(1);
                else
                {
                    local_in.push_back(-1);
                }
            }

            for(int k = 0;k<numOutputs;k++)
            {
                if(k<state[2])
                    local_out.push_back(1);
                else
                {
                    local_out.push_back(-1);
                }
            }
            for(int k = 0;k<local_in.size();k++)
            {
                cout << local_in[k] <<" ";
            }
            cout << endl;
            for(int k = 0;k<local_out.size();k++)
            {
                cout << local_out[k] <<" ";
            }
            cout << endl <<endl;;

            in.push_back(local_in);
            out.push_back(local_out);
            sampleSize++;
        }
    }
}


typedef void (*Learneble) (int &numInputs, int &numOutputs, int &sampleSize, string &type, vector<vector<bit_signature> > &in, vector<vector<bit_signature> > &out);


pair<string, Learneble /**for 0<i<15*/> Functions[8] =
        {
                //mp("a_i_is_a_i_plus_one_for_all", a_i_is_a_i_plus_one_for_all),
                //mp("longestSubstring_dai_di_is_0", longestSubstring_dai_di_is_0),
                mp("longestSubstring_ai_is_1", longestSubstring_ai_is_1),
                //mp("input_is_output", input_is_output),
                //mp("vector_loop_language", vector_loop_language),
                //mp("longest_substring_of_two_strings", longest_substring_of_two_strings),
                //mp("longest_substring_double_to_middle", longest_substring_double_to_middle),
                mp("end", nullFunction)
        };


void Data::generateData(int &ret_numInputs, int &ret_numOutputs, string &ret_type)
{

    clear();


    type = ret_type;

    int functionId = 0;

    while(Functions[functionId].f!="end")
    {
        if(Functions[functionId].f == type)
        {
            vector<vector<bit_signature> > proto_in;
            vector<vector<bit_signature> > proto_out;
            int proto_sample_size;
            int proto_numOutputs;
            int proto_numInputs = ret_numInputs;
            Functions[functionId].s(proto_numInputs, proto_numOutputs, proto_sample_size, type, proto_in, proto_out);
            assert(proto_in.size() == proto_out.size());
            for(int i = 0;i<proto_in.size(); i++)
            {
                push_back(proto_in[i], proto_out[i]);
            }
        }
        functionId++;
    }
    assert(in.size()==out.size());

    ret_type = type;
    ret_numInputs = numInputs;
    ret_numOutputs = numOutputs;
}

