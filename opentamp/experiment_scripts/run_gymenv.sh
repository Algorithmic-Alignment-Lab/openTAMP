#for N in 1 2 3 4 5
#do
#for S in third
#do

#python3 -W ignore policy_hooks/run_training.py -c policy_hooks.robosuite.jnt_pick_hyp \

python3 -W ignore policy_hooks/run_training.py -c new_specs.pointer_domain.new_env_hyperparam \
                                                -no 1 -llus 1000  -hlus 1000 \
                                                -spl -mask -hln 2 -lln 2 -hldim 256 -lldim 256 \
                                                -retime -vel 0.3 -eta 5 -softev \
                                                -lr_schedule fixed \
                                                -imwidth 256 -imheight 256 \
                                                -hist_len 2 -prim_first_wt 20 -lr 0.0001 \
                                                -hllr 0.001 -lldec 0.0001 -hldec 0.0004 \
                                                -add_noop 2 --permute_hl 1 \
                                                -expl_wt 10 -expl_eta 4 \
                                                -col_coeff 0.0 \
                                                -motion 4 \
                                                -n_gpu 1 \
                                                -rollout 0 \
                                                -task 1 \
                                                -post -pre \
                                                -warm 100 \
                                                -neg_ratio 0. -opt_ratio 0.9 -dagger_ratio 0.1 \
						-descr pointer_pilot_shorttrain_deterministicstart_richcontrolfeat
# sleep 1800 
# pkill -f run_train -9
# pkill -f ros -9
# sleep 5


#done
#done

