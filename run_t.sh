#! /bin/bash
result_path=./data/CTR_results
data_path=./data
for data_name in cf-t
do
  for ((k=1;k<=50;k=k+2))
  do
    for ((i=1;i<=20;i=i+1))
    do
    {
      ./lda-c/lda est 0.1 $k ./lda-c/settings.txt $data_path/$data_name/$data_name-mult.dat  random $result_path/$data_name-$k-$i
      ./ctr --directory $result_path/$data_name-$k-$i --user $data_path/$data_name/$data_name-train-$i-users.dat \
            --item $data_path/$data_name/$data_name-train-$i-items.dat --a 1 --b 0.01 --lambda_u 0.01 --lambda_v 100 \
            --mult $data_path/$data_name/$data_name-mult.dat --theta_init $result_path/$data_name-$k-$i/final.gamma \
            --beta_init $result_path/$data_name-$k-$i/final.beta \
            --num_factors $k --save_lag 20 --theta_opt

    }&
    done
    wait
  done
  wait
done

