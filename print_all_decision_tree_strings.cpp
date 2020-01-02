//
// Created by Kliment Serafimov on 2020-01-02.
//

#include "print_all_decision_tree_strings.h"

string print_decision_tree_string(string decision_tree_string,  int num_tabs, char open, char close, bool do_print)
{
    string tabbed_string;
    for(int i = 0;i<decision_tree_string.size();i++)
    {
        tabbed_string += get_tab(num_tabs);
        cout << get_tab(num_tabs);
        while(decision_tree_string[i] != open && decision_tree_string[i] != close && i < decision_tree_string.size())
        {
            tabbed_string += decision_tree_string[i];
            cout << decision_tree_string[i];
            i++;
        }
        if(decision_tree_string[i] == open)
        {
            tabbed_string += "\n";
            cout << "\n";

            if(do_print) {
                tabbed_string += get_tab(num_tabs) + decision_tree_string[i] + "\n";
                cout << get_tab(num_tabs) << decision_tree_string[i] << "\n";
            }
            num_tabs++;
        }
        else if(decision_tree_string[i] == close)
        {
            tabbed_string += "\n";
            cout << "\n";
            num_tabs--;

            if(do_print) {
                tabbed_string += get_tab(num_tabs) + decision_tree_string[i] + "\n";
                cout << get_tab(num_tabs) << decision_tree_string[i] << endl;
            }
        }
    }
    //fout_dataset << tabbed_string << endl;
    return tabbed_string;
}

string print_decision_tree_string_python(string decision_tree_string,  int num_tabs, char open, char close)
{
    string tabbed_string;
    for(int i = 0;i<decision_tree_string.size();i++)
    {
        tabbed_string += get_tab(num_tabs);
        cout << get_tab(num_tabs);
        while(decision_tree_string[i] != open && decision_tree_string[i] != close && i < decision_tree_string.size())
        {
            tabbed_string += decision_tree_string[i];
            cout << decision_tree_string[i];
            i++;
        }
        if(decision_tree_string[i] == open)
        {

            tabbed_string += "\n";
            cout << "\n";
//            tabbed_string += get_tab(num_tabs) + decision_tree_string[i] + "\n";
//            cout << get_tab(num_tabs) << decision_tree_string[i] << "\n";
            num_tabs++;
        }
        else if(decision_tree_string[i] == close)
        {
            tabbed_string += "\n";
            cout << "\n";
            num_tabs--;
//            tabbed_string += get_tab(num_tabs) + decision_tree_string[i] + "\n";
//            cout << get_tab(num_tabs) << decision_tree_string[i] <<endl;
        }
    }
    //fout_dataset << tabbed_string << endl;
    return tabbed_string;
}

string print_f(int n, int function)
{
    string tabbed_string;
    for(int i = 0; i <(1<<n);i++)
    {
        tabbed_string += toBinaryString(i, n) + "\n" + to_string(((function & (1<<i))  != 0)) + "\n";
        cout << toBinaryString(i, n) <<" " << ((function & (1<<i))  != 0) <<endl;
        //cout << toBinaryString(f, (1<<n)) <<endl;
    }
    return tabbed_string;
}

string print_decision_tree_of_f_cpp(int n, int func)
{
    cout << "here in print_decision_tree_of_f_cpp" <<endl;
    DecisionTreeScore score = _get_opt_decision_tree_score(n, func);
    string tabbed_string;
    assert(score.decision_tree_strings.size() == score.if_cpp_format_strings.size());
    for(int i = 0;i<score.decision_tree_strings.size();i++) {
        string local_str = score.if_cpp_format_strings[i];//"(dt=" + score.decision_tree_strings[i] + ")";
        tabbed_string+=print_decision_tree_string(local_str, 0, '{', '}', true);
        tabbed_string+="\n";
        cout << "\n";
    }
    tabbed_string+="\n";
    cout << "\n";
    //fout_dataset << tabbed_string << endl;
    return tabbed_string;
}

//string print_decision_tree_of_f_python(int n, int func)
//{
//    string tabbed_string;
//    cout << "here in print_decision_tree_of_f_python" <<endl;
//    DecisionTreeScore score = _get_opt_decision_tree_score(n, func);
//    assert(score.decision_tree_strings.size() == score.if_cpp_format_strings.size());
//    for(int i = 0;i<score.decision_tree_strings.size();i++) {
//        string local_str = score.if_python_format_strings[i];//"(dt=" + score.decision_tree_strings[i] + ")";
//        tabbed_string+=print_decision_tree_string_python(local_str, 0, '{', '}');
//        tabbed_string+="\n";
//        cout << "\n";
//    }
//    tabbed_string+="\n";
//    cout << "\n";
//    //fout_dataset << tabbed_string << endl;
//    return tabbed_string;
//}

void print_all_decision_tree_strings(int n)
{
    string tabbed_string = "";
    assert(n <= 4);
    for(int i = 0; i < (1<<(1<<n)); i++)
    {
        tabbed_string += "File file = File(name=\"out_to_str " + toBinaryString(i, (1<<n)) + "\",";
        tabbed_string += "dataset=\"";
        //tabbed_string += print_f(n, i);
        tabbed_string += "\")\n";
        tabbed_string += "File file = File(name=\"decision_tree_of_f " + toBinaryString(i, (1<<n)) + "\",";
        tabbed_string += "dataset=\"";

        int func = i;
        cout << "here in print_decision_tree_of_f_python" <<endl;

        Data local_new_data = Data();
        local_new_data.init_exaustive_table_with_unary_output(n, func);

        DecisionTreeSynthesisViaDP<Data> decision_tree_solver = DecisionTreeSynthesisViaDP<Data>();
        DecisionTreeScore size_opt;
        NeuralNetwork::parameters cutoff_parametes;
        size_opt = decision_tree_solver.synthesize_decision_tree_and_get_size
                (cutoff_parametes, n, local_new_data, optimal);

        DecisionTreeScore score = size_opt;

        string local_tabbed_string;
        assert(score.decision_tree_strings.size() == score.if_python_format_strings.size());
        for(int i = 0;i<score.decision_tree_strings.size();i++) {
            string local_str = score.if_python_format_strings[i];//"(dt=" + score.decision_tree_strings[i] + ")";
            local_tabbed_string+=print_decision_tree_string(local_str, 0, '{', '}', false);
            local_tabbed_string+="\n";
            cout << "\n";
        }

        local_tabbed_string+="\n";
        cout << "\n";

        tabbed_string += local_tabbed_string + "\")\n";
        tabbed_string += "-------------------";
        cout << "-------------------" << endl;
    }
    //fout_dataset << tabbed_string << endl;
}