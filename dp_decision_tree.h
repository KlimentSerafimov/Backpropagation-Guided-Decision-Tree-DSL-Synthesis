//
// Created by Kliment Serafimov on 2019-05-21.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DP_DECISION_TREE_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DP_DECISION_TREE_H

#include "Data.h"

template<typename datatype>
class dp_decision_tree
{
public:
    struct dp_ret
    {
        bool vis;

        int w;
        int w_root;



        int h;
        int h_root;

        vector<string> dt_strings;
        int num_solutions;

        void init(int _w, int _h, int _w_root, int _h_root, int _num_solutions)
        {
            vis = true;
            w = _w,
            h = _h,
            w_root = _w_root,
            h_root = _h_root,
            num_solutions = _num_solutions;
        }

        void init()
        {
            vis = false;
            w = (1<<30);
            h = (1<<30);
            num_solutions = 0;
        }

        void update(dp_ret left, dp_ret right, int mask_used, int track_num_strings)
        {
            assert(track_num_strings == -1);

            bool update_dt_strings = false;
            if(left.w + right.w + 1 < w)
            {
                w = left.w + right.w+1;
                w_root = mask_used;


                h = max(left.h, right.h)+1;
                h_root = mask_used;

                num_solutions = left.num_solutions*right.num_solutions;
                update_dt_strings = true;
                dt_strings.clear()
                assert(num_solutions < (1<<15));
            }
            else if(left.w + right.w + 1 == w)
            {
                num_solutions += left.num_solutions*right.num_solutions;
                update_dt_strings = true;

                assert(num_solutions < (1<<15));
            }

            if(update_dt_strings)
            {

                for(int i = 0; i < left.dt_strings; i++)
                {
                    for(int j = 0; j < right.dt_strings; j++)
                    {
                        dt_strings.pb("(mask_used="+to_string(mask_used) + " left=" + left[i] + " right=" + right[i]);
                    }
                }
            }
            /*if(max(left.h, right.h)+1 < h)
            {
                h = max(left.h, right.h)+1;
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
//            if(w<dp_w[mask])
//            {
//                dp_w[mask] = w;
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
            string ret = "w = " + to_string(w) + " h = "+ to_string(h);
            return ret;
        }
    };

//    bool vis[(1<<16)];
//
//    int dp_w[(1<<16)];
//    int dp_w_root[(1<<16)];
//
//    int dp_h[(1<<16)];
//    int dp_h_root[(1<<16)];
//
//    int dp_num_solutions[(1<<16)];

    dp_ret dp_data[1<<16];


    dp_ret rek(int n, datatype *the_data, int mask)
    {
        assert(mask < (1<<16));
        if(!dp_data[mask].vis)
        {
            dp_data[mask].vis = true;
            if(the_data->is_constant())
            {
                bit_signature out  = 0;
                for(int i = 0;i<the_data->size();i++)
                {
                    if(i == 0)
                        out = the_data->out[i][0];
                    else
                    {
                        assert(out == the_data->out[i][0]);
                    }
                }
                dp_data[mask].init(1, 1, mask, mask, 1);
            }
            else
            {
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

                    dp_ret left_ret = rek(n, &left, left_mask);
                    dp_ret right_ret = rek(n, &right, right_mask);

                    ret.update(left_ret, right_ret, left_mask, -1);

                }

                /*vector<pair<int, int> > active_pairs;
                for(int i = 0;i<active_nodes.size();i++)
                {
                    for(int j = i+1;j<active_nodes.size(); j++)
                    {
                        active_pairs.push_back(mp(active_nodes[i], active_nodes[j]));
                    }
                }

                vector<operator_signature> operators = the_data->get_operators(active_pairs);
                */
                //ret.update(mask, dp_w, dp_w_root, dp_h, dp_h_root, dp_num_solutions);
                dp_data[mask] = ret;
            }
        }

        return dp_data[mask];
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

        if(1 == dp_w[mask]) return;
        print_tree(n, dp_w_root[mask], t+1);
        print_tree(n, mask-dp_w_root[mask], t+1);
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


    DecisionTreeScore synthesize_decision_tree_and_get_size
            (net::parameters param, int n, datatype the_data, DecisionTreeSynthesiserType synthesizer_type)
    {
        DecisionTreeScore ret = DecisionTreeScore();

        if(synthesizer_type == optimal)
        {

            memset(vis, 0, sizeof(vis));
            memset(dp_w, 63, sizeof(dp_w));
            memset(dp_h, 63, sizeof(dp_h));

            if(n <= 4)
            {
                dp_ret opt;

                opt = rek(n, &the_data, (1<<(1<<n))-1);

                ret.size = opt.w;
                ret.num_solutions = opt.num_solutions;
                ret.dt_strings = opt.dt_strings;
            }
            else
            {
                assert(0);
            }

        }
        else
        {
            param.decision_tree_synthesiser_type = synthesizer_type;
            decision_tree extracted_tree = decision_tree(&the_data, param);
            //extracted_tree.print_gate(0);
            ret.size = extracted_tree.size;
        }
        return ret;
    }


    /*DecisionTreeScore old_extract_decision_tree_and_compare_with_opt_and_entropy(net::parameters param, int n, Data the_data)

    {
        param.decision_tree_synthesiser_type = confusion_guided;
        decision_tree confusion_extracted_tree = decision_tree(&the_data, param);
        int confusion_guided_size = confusion_extracted_tree.size;

        param.decision_tree_synthesiser_type = entropy_guided;
        decision_tree entropy_based_tree = decision_tree(&the_data, param);
        int entropy_guided_size = entropy_based_tree.size;

        DecisionTreeScore ret;
        dp_ret opt;

        if(n <= 4)
        {
            opt = rek(n, &the_data, (1<<(1<<n))-1);
            ret.set_opt(opt.w);
        }

        ret.neural_guided_size = confusion_guided_size;
        ret.entropy_guided_size = entropy_guided_size;

        return ret;
    }

    DecisionTreeScore new_extract_decision_tree_and_compare_with_opt_and_entropy(net::parameters param, int n, Data the_data)
    {
        memset(vis, 0, sizeof(vis));
        memset(dp_w, 63, sizeof(dp_w));
        memset(dp_h, 63, sizeof(dp_h));

        param.decision_tree_synthesiser_type = neural_guided;
        decision_tree neurally_extracted_tree = decision_tree(&the_data, param);
        int neural_guided_size = neurally_extracted_tree.size;

        param.decision_tree_synthesiser_type = entropy_guided;
        decision_tree entropy_based_tree = decision_tree(&the_data, param);
        int entropy_guided_size = entropy_based_tree.size;

        DecisionTreeScore ret;
        dp_ret opt;

        if(n <= 4)
        {
            opt = rek(n, &the_data, (1<<(1<<n))-1);
            ret.set_opt(opt.w);
        }
        ret.neural_guided_size = neural_guided_size;
        ret.entropy_guided_size = entropy_guided_size;

        return ret;
    }*/

    void dp_init(int n, datatype the_data)
    {
        memset(vis, 0, sizeof(vis));
        memset(dp_w, 63, sizeof(dp_w));
        memset(dp_h, 63, sizeof(dp_h));
        dp_ret ret = rek(n, &the_data, (1<<(1<<n))-1);

        //the_data.printData("data");

        //cout << "tree" <<endl;
        //print_tree((1<<16)-1, 0);

        //cout <<"score = " << ret.print() <<endl;

        int id_iter_count = 8;//8;
        int id_ensamble_size = 10;

        net::parameters param = net::parameters(1, 1);
        param.set_number_resets(1);
        param.num_stale_iterations = 10000;
        param.set_iteration_count(id_iter_count);
        param.ensamble_size = id_ensamble_size;
        param.priority_train_f = &net::softPriorityTrain;//&net::harderPriorityTrain;
        param.first_layer = 2*n;
        param.ensamble_size = 1;

        //try to pick dimension that maximizes confusion differnece between the two segments;

        int heuristic_ret = 100;
        decision_tree best_tree = decision_tree();
        for(int i = 1;i<=1;i++)
        {
            decision_tree tree = decision_tree(&the_data, param);
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
            if(heuristic_ret-ret.w >= 2)
            {
                the_data.printData(to_string(heuristic_ret-ret.w)+" errors");
                print_tree(n, (1<<(1<<n))-1, 0);
                best_tree.print_gate(0);
            }
            if(false && ret.w == 11 && heuristic_ret-ret.w == 0)
            {
                the_data.printData("0 errors");
                print_tree(n, (1<<(1<<n))-1, 0);

                best_tree.print_gate(0);
            }
        }
        //cout << "end" <<endl;
        rez[ret.w]+=heuristic_ret;
        num_rez[ret.w]++;
        num_correct[ret.w]+=(heuristic_ret == ret.w);
        num_wrong[ret.w][heuristic_ret-ret.w]++;
        cout << ret.w <<" "<< (double)rez[ret.w]/num_rez[ret.w] <<endl;
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

#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_DP_DECISION_TREE_H
