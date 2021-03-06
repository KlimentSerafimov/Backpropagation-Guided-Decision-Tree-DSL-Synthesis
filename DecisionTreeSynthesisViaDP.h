//
// Created by Kliment Serafimov on 2019-05-21.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DP_DECISION_TREE_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DP_DECISION_TREE_H

#include "Data.h"
#include "util.h"
#include "NeuralNetwork.h"
#include "DecisionTree.h"
#include "FunctionAndDecisionTreeScore.h"

static int era_counter = 0;

struct dp_ret
{
    //bool vis;

    int w;
    int mask;

    int h;

    vector<string> decision_tree_strings;
    vector<string> if_cpp_format_strings;
    vector<string> if_python_format_strings;

    vector<pair<int, pair<dp_ret*, dp_ret*> > > children;

    int num_solutions;

    int local_era = -1;

    bool get_vis(int local_era_counter)
    {
//        cout << "Get_vis = " << local_era <<" "<< local_era_counter << endl;
        return local_era == local_era_counter;
    }

    void set_vis(int local_era_counter)
    {
//        cout << "set_vis = " << local_era_counter << endl;
        local_era = local_era_counter;
    }

    void set_vis_false()
    {
        local_era = -1;
    }

    int mask_size = -1;

    string mask_to_string()
    {
        if(mask >= 0)
        {
            return " mask=" + toBinaryString(mask, mask_size);
        }
        else
        {
            return " is_leaf";
        }
    }

    string state_to_string()
    {
        string bracket = "(";
        string name = bracket +
                      "weights=" + to_string(w) +
                      " h=" + to_string(h) +
                      mask_to_string() +
                      " num_solutions=" + toDecimalString(num_solutions) +
                      ")";
        return name;
    }

    void init(int _mask_size, int _w, int _h, int _num_solutions, int out, int local_era_counter)
    {
        mask_size = _mask_size;
        set_vis(local_era_counter);
        w = _w,
        h = _h,

        mask = -1;

        num_solutions = _num_solutions;
        decision_tree_strings.push_back(state_to_string());
        if_cpp_format_strings.push_back("return " + to_string(out) + ";");
        if_python_format_strings.push_back("return " + to_string(out));
    }

    void init()
    {
        set_vis_false();
        w = (1<<30);
        h = (1<<30);
        mask = -2;
        num_solutions = 0;
        if_cpp_format_strings.clear();
        if_python_format_strings.clear();
        decision_tree_strings.clear();
    }

    dp_ret()
    {
        init();
    }

    void update(dp_ret* left, dp_ret* right, int mask_used, int _input_id, int track_num_strings)
    {
        assert(left->mask_size == right->mask_size);
        assert(track_num_strings == -1);

        mask_size = left->mask_size;

        bool update_decision_tree_strings = false;
        if(left->w + right->w + 1 < w)
        {
            w = left->w + right->w+1;
            mask = mask_used;


            h = max(left->h, right->h)+1;

            num_solutions = left->num_solutions*right->num_solutions;
            update_decision_tree_strings = true;
            decision_tree_strings.clear();

            children.clear();
            if_cpp_format_strings.clear();
            if_python_format_strings.clear();

            assert(num_solutions < (1<<15));
        }
        else if(left->w + right->w + 1 == w)
        {
            num_solutions += left->num_solutions*right->num_solutions;
            update_decision_tree_strings = true;

            assert(num_solutions < (1<<15));
        }

        if(update_decision_tree_strings)
        {

            assert(if_python_format_strings.size() == if_cpp_format_strings.size());
            assert(decision_tree_strings.size() == if_cpp_format_strings.size());
            assert(left->decision_tree_strings.size() == left->if_cpp_format_strings.size());
            assert(right->decision_tree_strings.size() == right->if_cpp_format_strings.size());
            for(int i = 0; i < left->decision_tree_strings.size(); i++)
            {
                for(int j = 0; j < right->decision_tree_strings.size(); j++)
                {
                    string bracket = "(";
                    string name = bracket +
                                  "state="+
                                  state_to_string()+
                                  " left=" + left->decision_tree_strings[i] +
                                  " right=" + right->decision_tree_strings[j] +
                                  ")";
                    //cout << name << endl;
                    decision_tree_strings.push_back(name);

                    string if_cpp_format_str =
                            "if(x["+to_string(_input_id)+ "]){" +
                            left->if_cpp_format_strings[i] + "}else{" + right->if_cpp_format_strings[j] + "}";

                    string if_python_format_str =
                            "if x["+to_string(_input_id)+ "] == 1:{" +
                            left->if_python_format_strings[i] + "}else:{" + right->if_python_format_strings[j] + "}";

                    if_cpp_format_strings.push_back(if_cpp_format_str);
                    if_python_format_strings.push_back(if_python_format_str);
                }
            }

            children.push_back(make_pair(_input_id, make_pair(left, right)));
        }
        /*if(max(left->h, right->h)+1 < h)
        {
            h = max(left->h, right->h)+1;
            h_root = mask_used;
        }*/
    }

//        void update(
//                int mask,
//                int *dp_w,
//                int *dp_w_root,
//
//                int *dp_h,
//                int *dp_h_root,
//
//                int *dp_num_solutions
//        )
//        {
//            if(weights<dp_w[mask])
//            {
//                dp_w[mask] = weights;
//                dp_w_root[mask] = w_root;
//
//                dp_h[mask] = h;
//                dp_h_root[mask] = h_root;
//
//                dp_num_solutions[mask] = num_solutions;
//            }
//            /*if(h<dp_h[mask])
//            {
//                dp_h[mask] = h;
//                dp_h_root[mask] = h_root;
//            }* /
//        }
    string print()
    {
        string ret = "weights = " + to_string(w) + " h = "+ to_string(h);
        return ret;
    }
};

static dp_ret dp_data[1<<16];

template<typename datatype>
class DecisionTreeSynthesisViaDP
{
public:

//    bool vis[(1<<16)];
//
//    int dp_w[(1<<16)];
//    int dp_w_root[(1<<16)];
//
//    int dp_h[(1<<16)];
//    int dp_h_root[(1<<16)];
//
//    int dp_num_solutions[(1<<16)];

    int local_era_counter;

    DecisionTreeSynthesisViaDP()
    {

    }


    dp_ret* rek(int n, datatype* the_data, int mask)
    {
        assert(mask < (1<<16));
        if(!dp_data[mask].get_vis(local_era_counter))
        {
//            cout << "enter " << local_era_counter << endl;
            dp_data[mask].init();
            dp_data[mask].set_vis(local_era_counter);
            if(the_data->is_constant())
            {

//                cout << "is_constant " << local_era_counter << endl;
                bit_signature out = 0;
                for(int i = 0;i<the_data->size();i++)
                {
                    if(i == 0)
                    {
                        out = the_data->out[i][0];
                    }
                    else
                    {
                        assert(out == the_data->out[i][0]);
                    }
                }
                dp_data[mask].init((1<<n), 1, 1, 1, out.value, local_era_counter);
            }
            else
            {
//                cout << "composite " << local_era_counter << endl;
                vector<int> active_nodes = the_data->get_active_bits();

                dp_ret ret;
                ret.init();

                for(int i = 0;i<active_nodes.size();i++)
                {
                    datatype left, right;
                    bit_signature split_idx = bit_signature(active_nodes[i]);

                    assert(split_idx.vector_id == active_nodes[i]);
                    the_data->split(split_idx, left, right);

                    int left_mask = 0;
                    int right_mask = 0;

                    for(int j = 0, in_row = 0;j<(1<<n);j++)
                    {
                        if((mask&(1<<j)) != 0)
                        {
                            assert(in_row < the_data->in.size());
                            assert(split_idx.vector_id < the_data->in[in_row].size());

                            if(the_data->in[in_row][split_idx.vector_id] == 1)
                            {
                                left_mask|=(1<<j);
                            }
                            else if(the_data->in[in_row][split_idx.vector_id] == -1)
                            {
                                right_mask|=(1<<j);
                            }
                            in_row++;
                        }
                    }

                    assert((left_mask|right_mask) == mask);
                    //cout << bitset<16>(left_mask).to_string() <<endl;
                    //cout << bitset<16>(right_mask).to_string() <<endl;
                    //cout << endl;

                    dp_ret* left_ret = rek(n, &left, left_mask);
                    dp_ret* right_ret = rek(n, &right, right_mask);

                    ret.update(left_ret, right_ret, left_mask, active_nodes[i], -1);

                }

                /*vector<pair<int, int> > active_pairs;
                for(int i = 0;i<active_nodes.size();i++)
                {
                    for(int j = i+1;j<active_nodes.size(); j++)
                    {
                        active_pairs.push_back(make_pair(active_nodes[i], active_nodes[j]));
                    }
                }

                vector<operator_signature> operators = the_data->get_operators(active_pairs);
                */
                //ret.update(mask, dp_w, dp_w_root, dp_h, dp_h_root, dp_num_solutions);
                dp_data[mask] = ret;
            }
        }

//        cout << "Exit  " << local_era_counter << endl;
        assert(dp_data[mask].mask_size == (1<<n));
        return &dp_data[mask];
    }

    void print_tree(int n, int mask, int t)
    {
        if(n == 4)
        {
            cout << indent(t) << bitset<16>(mask).to_string() <<endl;
        }
        else if(n == 3)
        {
            cout << indent(t) << bitset<8>(mask).to_string() <<endl;
        }
        else if(n == 2)
        {

            cout << indent(t) << bitset<4>(mask).to_string() <<endl;
        }
        else
        {
            assert(0);
        }

        if(1 == dp_data[mask].w) return;
        print_tree(n, dp_data[mask].mask, t+1);
        print_tree(n, mask-dp_data[mask].mask, t+1);
    }

    int rez[33];
    int num_rez[33];
    int num_correct[33];
    int num_wrong[33][11];

    //6.100
    //6.890
    //6.s081
    //6.033
    //6.UAT

    //6.s898
    //CMS.333

    //IOI with Vlade, Physics with Martin, intro to programming with Josif
    //IDEAS global challenge
    //Video for Fond for innovation

    void initialize_local_optimal_search()
    {
        local_era_counter = era_counter+1;
//        cout << "initialize " << era_counter+1 << endl;
        era_counter++;
    }

    DecisionTreeScore synthesize_decision_tree_and_get_size
            (NeuralNetwork::parameters param, int n, datatype the_data, DecisionTreeSynthesiserType synthesizer_type)
    {
        DecisionTreeScore ret = DecisionTreeScore();

        if(synthesizer_type == optimal)
        {
            if(n <= 4)
            {
                dp_ret* opt;

                initialize_local_optimal_search();

                opt = rek(n, &the_data, (1<<(1<<n))-1);

                ret.size = opt->w;
                ret.num_solutions = opt->num_solutions;
                ret.decision_tree_strings = opt->decision_tree_strings;
                ret.if_cpp_format_strings = opt->if_cpp_format_strings;
                ret.if_python_format_strings = opt->if_python_format_strings;
            }
            else
            {
                assert(0);
            }

        }
        else
        {
            param.decision_tree_synthesiser_type = synthesizer_type;
            DecisionTree extracted_tree = DecisionTree(&the_data, param);
            //extracted_tree.print_gate(0);
            ret.size = extracted_tree.size;
        }
        return ret;
    }


    /*DecisionTreeScore old_extract_decision_tree_and_compare_with_opt_and_entropy(NeuralNetwork::parameters param, int n, Data the_data)

    {
        param.decision_tree_synthesiser_type = confusion_guided;
        DecisionTree confusion_extracted_tree = DecisionTree(&the_data, param);
        int confusion_guided_size = confusion_extracted_tree.size;

        param.decision_tree_synthesiser_type = entropy_guided;
        DecisionTree entropy_based_tree = DecisionTree(&the_data, param);
        int entropy_guided_size = entropy_based_tree.size;

        DecisionTreeScore ret;
        dp_ret opt;

        if(n <= 4)
        {
            opt = rek(n, &the_data, (1<<(1<<n))-1);
            ret.set_opt(opt.weights);
        }

        ret.neural_guided_size = confusion_guided_size;
        ret.entropy_guided_size = entropy_guided_size;

        return ret;
    }

    DecisionTreeScore new_extract_decision_tree_and_compare_with_opt_and_entropy(NeuralNetwork::parameters param, int n, Data the_data)
    {
        memset(vis, 0, sizeof(vis));
        memset(dp_w, 63, sizeof(dp_w));
        memset(dp_h, 63, sizeof(dp_h));

        param.decision_tree_synthesiser_type = neural_guided;
        DecisionTree neurally_extracted_tree = DecisionTree(&the_data, param);
        int neural_guided_size = neurally_extracted_tree.size;

        param.decision_tree_synthesiser_type = entropy_guided;
        DecisionTree entropy_based_tree = DecisionTree(&the_data, param);
        int entropy_guided_size = entropy_based_tree.size;

        DecisionTreeScore ret;
        dp_ret opt;

        if(n <= 4)
        {
            opt = rek(n, &the_data, (1<<(1<<n))-1);
            ret.set_opt(opt.weights);
        }
        ret.neural_guided_size = neural_guided_size;
        ret.entropy_guided_size = entropy_guided_size;

        return ret;
    }*/

    void dp_init(int n, datatype the_data)
    {
        initialize_local_optimal_search();
//        memset(vis, 0, sizeof(vis));
//        memset(dp_w, 63, sizeof(dp_w));
//        memset(dp_h, 63, sizeof(dp_h));
        dp_ret* ret = rek(n, &the_data, (1<<(1<<n))-1);

        //the_data.printData("data");

        //cout << "tree" <<endl;
        //print_tree((1<<16)-1, 0);

        //cout <<"score = " << ret.print() <<endl;

        int id_iter_count = 8;//8;
        int id_ensamble_size = 10;

        NeuralNetwork::parameters param = NeuralNetwork::parameters(1, 1);
        param.set_number_resets(1);
        param.num_stale_iterations = 10000;
        param.set_iteration_count(id_iter_count);
        param.ensamble_size = id_ensamble_size;
        param.priority_train_f = &NeuralNetwork::softPriorityTrain;//&NeuralNetwork::harderPriorityTrain;
        param.first_layer = 2*n;
        param.ensamble_size = 1;
        param.decision_tree_synthesiser_type = confusion_guided;

        //try to pick dimension that maximizes confusion differnece between the two segments;

        int heuristic_ret = 100;
        DecisionTree best_tree = DecisionTree();
        for(int i = 1;i<=1;i++)
        {
            DecisionTree tree = DecisionTree(&the_data, param);
            if(tree.size<heuristic_ret)
            {
                best_tree = tree;
                heuristic_ret = tree.size;
            }
            //heuristic_ret = min(heuristic_ret, tree.size);
            //cout << tree.size <<" "<< tree.height <<endl;
        }
        if(false || print_close_local_data_model)
        {
            if(heuristic_ret-ret->w >= 2)
            {
                the_data.printData(to_string(heuristic_ret-ret->w)+" errors");
                print_tree(n, (1<<(1<<n))-1, 0);
                best_tree.print_gate(0);
            }
            if(false && ret->w == 11 && heuristic_ret-ret->w == 0)
            {
                the_data.printData("0 errors");
                print_tree(n, (1<<(1<<n))-1, 0);

                best_tree.print_gate(0);
            }
        }
        //cout << "end" <<endl;
        rez[ret->w]+=heuristic_ret;
        num_rez[ret->w]++;
        num_correct[ret->w]+=(heuristic_ret == ret->w);
        num_wrong[ret->w][heuristic_ret-ret->w]++;
        cout << ret->w <<" "<< (double)rez[ret->w]/num_rez[ret->w] <<endl;
    }

    void run()
    {
        memset(rez, 0, sizeof(rez));
        memset(num_rez, 0, sizeof(num_rez));
        memset(num_correct, 0, sizeof(num_correct));
        memset(num_wrong, 0, sizeof(num_wrong));

        int unit_calc = 29;//495;
        vector<int> unit_calcs;
        //unit_calcs.push_back((1<<8)-1);
        //unit_calcs.push_back((1<<16)-1-(1<<4)-(1<<8));
        unit_calcs.push_back(unit_calc);
        unit_calcs.push_back(unit_calc);
        unit_calcs.push_back(unit_calc);
        unit_calcs.push_back(unit_calc);
        /*for(int i = 0;i<16;i++)
        {
            unit_calcs.push_back(unit_calc^(1<<i));
        }*/

        int n = 3;

        for(int i = 0;i<1000*0+1*(1<<(1<<n));i+=1+0*rand(0, 120))
            //for(int i = unit_calc;i<=unit_calc;i+=1+0*rand(0, 120))
            //for(int at = 0, i = unit_calcs[at]; at<unit_calcs.size();at++,  i = unit_calcs[at])
        {
            datatype new_data;

            new_data.init_exaustive_table_with_unary_output(n, i);
            //new_data.printData("init");
            cout << i<<" ";
            dp_init(n, new_data);
            /*
            if(i%100 == 0)
            {
                print_score();
            }*/
        }
        print_score();
    }

    void print_score()
    {
        for(int i = 1;i<31;i+=2)
        {
            cout << i <<"\t"<< (double)rez[i]/num_rez[i]  << "\t" << ((double)rez[i]/num_rez[i])/i << endl;
        }
        int sum_train_iter_correct = 0;
        int sum_train_iter = 0;
        for(int i = 1;i<31;i+=2)
        {
            cout << i <<"\t"<<"::"<<"\t";
            int local_correct = 0;
            int local_sum_train_iter = 0;
            for(int j = 0;j<=12;j+=2)
            {
                int num = num_wrong[i][j];
                if(j == 0)
                {
                    sum_train_iter_correct+=num;
                    local_correct+=num;
                }
                sum_train_iter+=num;
                local_sum_train_iter+=num;

                if(num == 0)
                {
                    cout << ".\t";
                }
                else
                {
                    cout << num <<"\t";
                }
            }
            cout << "correct :: \t" << local_correct << "\t" << local_sum_train_iter;
            cout << endl;
        }
        cout << "correct = " << sum_train_iter_correct <<"/"<<sum_train_iter <<endl;
    }
};

DecisionTreeScore get_opt_decision_tree_score(int n, int func);

vector<FunctionAndDecisionTreeScore> get_smallest_f(int n);

#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DP_DECISION_TREE_H
