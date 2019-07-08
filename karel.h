//
// Created by Kliment Serafimov on 2019-02-20.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_KAREL_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_KAREL_H


#include "Header.h"

class karel_type
{
public:
    class cell_type
    {
    public:
        int count;
        cell_type(int _count)
        {
            count = _count;
        }
        cell_type& operator ++ ()
        {
            count++;
            return *this;
        }
    };
    class hero_type
    {
    public:
        int loc;
        int dir;
    };

    hero_type hero;

    vector<cell_type> mat;

    karel_type(int n)
    {
        assert(n%2 == 1);
        hero.loc = n/2;
        hero.dir = 1;
        mat = vector<cell_type>(n, 0);
    }
};

class karel_interpreter
{
    enum primitive {k_move, k_rotate, k_putMarker};

    vector<primitive> code;

    karel_interpreter()
    {

    }

    void move(karel_type *board)
    {
        assert(board->hero.loc >= 0 && board->hero.loc < board->mat.size());
        board->hero.loc+=board->hero.dir;
        assert(board->hero.loc >= 0 && board->hero.loc < board->mat.size());
    }

    void rotate(karel_type *board)
    {
        assert(board->hero.dir == 1 || board->hero.dir == -1);
        board->hero.dir*=-1;
        assert(board->hero.dir == 1 || board->hero.dir == -1);
    }
    
    void putMarker(karel_type *board)
    {
        assert(board->hero.loc >= 0 && board->hero.loc < board->mat.size());
        board->mat[board->hero.loc]++;
    }

    void run(karel_type *board)
    {
        for(int i = 0;i<code.size();i++)
        {
            if(code[i] == k_move)
            {
                move(board);
            }
            else if(code[i] == k_rotate)
            {
                rotate(board);
            }
            else if(code[i] == k_putMarker)
            {
                putMarker(board);
            }
            else
            {
                assert(0);
            }
        }
    }

    void make_random_code(int n)
    {
        for(int i = 0;i<n;i++)
        {
            code.pb(static_cast<primitive>(rand(0, 3)));
        }
    }
};

#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_KAREL_H
