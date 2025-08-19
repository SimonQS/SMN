#!/bin/bash
# chmod u+x exp_1.sh
# ./exp_1.sh

# Define the save_variable function
echo "Start"
save_variable() {
    target="$1"
    name="$2"

    dirname="$(dirname "$0")"
    filename="$dirname/results_new/$name"

    echo "$target" >> "$filename"  # Append results to the file
    echo "Variable saved!"
}
echo "1"
citation=("cora" "citeseer" "pubmed") 
fb_num=(0 107 348 414 686)
mag=("mag_eng" "mag_cs" "mag_chem" "mag_med" )
cs=("sub_cs" "sub_topk")
case=(1 2)

# Initialize arrays for storing data
# declare -A exp1_disjoint
declare -A exp_smn_nmi

# EXP 1: cs-test
# exp1_disjoint="cs,dataset,Training Time,Query Time,Test_acc/f1,Accuracy,F1 Score"
# save_variable "$exp1_disjoint" "exp1_disjoint.csv"
exp_smn_nmi="cs,case,dataset,Training Time,Query Time,Test_acc/f1,NMI,Jaccard,F1 Score"
save_variable "$exp_smn_nmi" "exp_smn_nmi.csv"

for i in "${cs[@]}"
do
    for j in "${citation[@]}"
    do
        # RUN SMN on three citation and Facebook
        echo "Running experiment with cs=$i and dataset=$j"

        # Modified experiment command with F1 score output
        citation_output=$(python citation.py --cs "$i" --dataset "$j")
        t_time=$(echo "$citation_output"  | grep "Training Time: " | awk '{print $NF}')
        q_time=$(echo "$citation_output" | grep "Query Time: " | awk '{print $NF}')
        test=$(echo "$citation_output" | grep "Test_acc/f1: " | awk '{print $NF}')
        acc_cs=$(echo "$citation_output" | grep "Accuracy: " | awk '{print $NF}')
        f1_score=$(echo "$citation_output" | grep "F1 Score: " | awk '{print $NF}')

        # Store the results in the data arrays
        exp1_disjoint="$i, $j, $t_time, $q_time, $test, $acc_cs, $f1_score"
        save_variable "$exp1_disjoint" "exp1_disjoint.csv"
    done

    echo "Running experiment with cs=$i for reddit"

    # Modified experiment command with F1 score output
    red_output=$(python reddit.py --cs "$i")        
    t_time=$(echo "$red_output"  | grep "Training Time: " | awk '{print $NF}')
    q_time=$(echo "$red_output" | grep "Query Time: " | awk '{print $NF}')
    test=$(echo "$red_output" | grep "Test_acc/f1: " | awk '{print $NF}')
    acc_cs=$(echo "$red_output" | grep "Accuracy: " | awk '{print $NF}')
    f1_score=$(echo "$red_output" | grep "F1 Score: " | awk '{print $NF}')

    # Store the results in the data arrays
    exp1_disjoint="$i, reddit, $t_time, $q_time, $test, $acc_cs, $f1_score"
    save_variable "$exp1_disjoint" "exp1_disjoint.csv"

    ######################## Overlapping ########################

    for k in "${fb_num[@]}"
    do
        # RUN SMN on Facebook
        for m in "${case[@]}"
        do
            echo "Running experiment with cs=$i and dataset=facebook$k and case=$m"

            # Modified experiment command with F1 score output
            facebook_output=$(python main.py --cs "$i" --dataset "facebook" --fb_num "$k" --case "$m" --loss "focal")
            t_time=$(echo "$facebook_output"  | grep "Training Time: " | awk '{print $NF}')
            q_time=$(echo "$facebook_output" | grep "Query Time: " | awk '{print $NF}')
            test=$(echo "$facebook_output" | grep "Test_acc/f1: " | awk '{print $NF}')
            nmi_cs=$(echo "$facebook_output" | grep "NMI: " | awk '{print $NF}')
            acc_cs=$(echo "$facebook_output" | grep "Jaccard: " | awk '{print $NF}')
            f1_score=$(echo "$facebook_output" | grep "F1 Score: " | awk '{print $NF}')

            # Store the results in the data arrays
            exp_smn_nmi="$i, $m, FB$k, $t_time, $q_time, $test, $nmi_cs, $acc_cs, $f1_score"
            save_variable "$exp_smn_nmi" "exp_smn_nmi.csv"
        done
    done

    for l in "${mag[@]}"
    do
        for m in "${case[@]}"
        do
            # RUN SMN on three citation and Facebook
            echo "Running experiment with cs=$i and dataset=$l and case=$m"

            # Modified experiment command with F1 score output
            mag_output=$(python main.py --cs "$i" --dataset "$l" --case "$m")        
            t_time=$(echo "$mag_output"  | grep "Training Time: " | awk '{print $NF}')
            q_time=$(echo "$mag_output" | grep "Query Time: " | awk '{print $NF}')
            test=$(echo "$mag_output" | grep "Test_acc/f1: " | awk '{print $NF}')
            nmi_cs=$(echo "$mag_output" | grep "NMI: " | awk '{print $NF}')
            acc_cs=$(echo "$mag_output" | grep "Jaccard: " | awk '{print $NF}')
            f1_score=$(echo "$mag_output" | grep "F1 Score: " | awk '{print $NF}')

            # Store the results in the data arrays
            exp_smn_nmi="$i, $m, $l, $t_time, $q_time, $test, $nmi_cs, $acc_cs, $f1_score"
            save_variable "$exp_smn_nmi" "exp_smn_nmi.csv"
        done
    done
done
