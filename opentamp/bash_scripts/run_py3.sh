#for N in 1 2 3 4 5
#do
#for S in third
#do


python3 -W ignore policy_hooks/run_training.py -c policy_hooks.namo.hyperparams_v94 \
                                                -no 1 -llus 10  -hlus 10 \
                                                -spl -mask -hln 2 -hldim 96 -lldim 64 \
                                                -retime -vel 0.3 -eta 5 -softev \
                                                -lr_schedule adaptive \
                                                -obs_del -hist_len 2 -prim_first_wt 20 -lr 0.0002 \
                                                -hllr 0.001 -lldec 0.0001 -hldec 0.0004 \
                                                -add_noop 2 --permute_hl 1 \
                                                -expl_wt 10 -expl_eta 4 \
                                                -col_coeff 0.0 \
                                                -motion 5 \
                                                -n_gpu 0 \
                                                -rollout 1 \
                                                -task 1 \
                                                -post -pre \
                                                -warm 100 \
                                                -neg_ratio 0. -opt_ratio 0.7 -dagger_ratio 0.3 \
                                                -test "namo_objs1_1/mac_test_49" \
                                                -index 49 \
                                                -descr mac_test & 
sleep 10800 
pkill -f run_train -9
pkill -f ros -9
sleep 5


#done
#done

