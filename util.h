//
// Created by Kliment Serafimov on 2019-02-16.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_UTIL_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_UTIL_H

#include "bit_signature.h"

static int printCycle = false;
static int printItteration = false;
static int printFullOrder = false;
static int printTopologicalOrder = false;
static int printMST = false;
static int printStateActionReward = false;
static int print_delta_knowledge_graph = false;
static int print_important_neurons = false;
static int print_classify_neurons = false;
static int print_implications = false;
static int print_the_imporant_bit = false;
static int printCleanItteration = false;
static int print_discrete_model = false;
static int printNewChildren = false;
static int print_tree_synthesys = false;
static int print_try_and_finally = false;
static int print_close_local_data_model = false;
static int printOnlyBatch = false;

vector<bit_signature> to_bit_signature_vector(int num_bits, int bit);

double sum_vector(vector<bit_signature> v);

double max_of_vector(vector<bit_signature> v);

vector<bit_signature> get_error_vector(vector<bit_signature> correct, vector<bit_signature> predict, int the_pow);

double get_max_of_error_vector(vector<bit_signature>* correct, vector<bit_signature>* predict, int the_pow);

bool check(vector<bit_signature> *correct, vector<bit_signature> *predict, double accuracy);

vector<bit_signature> vector_value_product(double value, vector<bit_signature> vec);

vector<bit_signature> pairwise_negation(vector<bit_signature> first, vector<bit_signature> second);

vector<bit_signature> pairwise_addition(vector<bit_signature> first, vector<bit_signature> second);

vector<bit_signature> pairwise_product(vector<bit_signature> first, vector<bit_signature> second);

vector<bit_signature> pairwise_division(vector<bit_signature> first, vector<bit_signature> second);

void cout_vector(vector<int> v);

string toString(int n, int k, int l);

string toBinaryString(int n, int l);

string toDecimalString(int n);

string indent(int n);

string get_tab(int num_tabs);

enum DecisionTreeSynthesiserType : int {undefined, optimal, neural_guided, confusion_guided, entropy_guided, combined_guided, random_guided};

#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_UTIL_H
