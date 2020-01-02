//
// Created by Kliment Serafimov on 10/8/18.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_HEADER_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_HEADER_H

//
//  Header.hpp
//  neural guided decision tree
//
//  Created by Kliment Serafimov on 6/3/18.
//  Copyright © 2018 Kliment Serafimov. All rights reserved.
//

#include <stdio.h>

#include <fstream>
#include <iomanip>
#include <iostream>

#include <map>
#include <set>
#include <cmath>
#include <queue>
#include <stack>
#include <math.h>
#include <time.h>
#include <string>
#include <vector>
#include <random>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <stdlib.h>
#include <algorithm>

#define P push
#define rand(a, b) ((rand()%(b-a+1))+a)
#define sort_v(a) sort(a.begin(), a.end())
#define rev_v(a)  reverse(a.begin(), a.end())

using namespace std;

const static int IS_GATE = 2;
const static int AND_GATE = 8;
const static int OR_GATE = 14;
const static int NAND_GATE = 7;
const static int NOR_GATE = 1;


#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_HEADER_H
