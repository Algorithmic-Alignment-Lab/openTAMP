#for N in 1 2 3 4 5
#do
#for S in third
#do

#python3 -W ignore policy_hooks/run_training.py -c policy_hooks.robosuite.jnt_pick_hyp \

# runs the main job

TRIAL_NAME='nav_dettrial_fullobs_bimodalwideavoid_noinit'

python3 -W ignore policy_hooks/run_training.py -c new_specs.nav_domain_belief_dettest.new_env_hyperparam \
                                                -no 1 -llus 1000  -hlus 1000 \
                                                -spl -mask -hln 2 -lln 2 -hldim 32 -lldim 32 \
                                                -retime -vel 0.3 -eta 5 -softev \
                                                -lr_schedule fixed \
                                                -imwidth 256 -imheight 256 \
                                                -hist_len 2 -prim_first_wt 20 -lr 0.001 \
                                                -hllr 0.0001 -contlr 0.0001 -lldec 0.0001 -hldec 0.0001 \
                                                --permute_hl 1 \
                                                -expl_wt 10 -expl_eta 4 \
                                                -col_coeff 0.0 \
                                                -motion 12 \
                                                -n_gpu 1 \
                                                -rollout 5 \
                                                -task 1 \
                                                -post -pre \
                                                -warm 1000 \
                                                -run_time 3600 \
                                                -neg_ratio 0. -opt_ratio 1.0 -dagger_ratio 0.0 \
						-descr $TRIAL_NAME 

# adds renders after job is done -- majority of options here ignored

python3 -W ignore -m cProfile -o profile_out policy_hooks/run_training.py -c new_specs.nav_domain_belief_dettest.new_env_hyperparam \
                                                -no 1 -llus 100  -hlus 100 \
                                                -spl -mask -hln 2 -hldim 256 -lldim 256 \
                                                -retime -vel 0.3 -eta 5 -softev \
                                                -lr_schedule adaptive \
                                                -imwidth 256 -imheight 256 \
                                                -hist_len 2 -prim_first_wt 20 -lr 0.00005 \
                                                -hllr 0.001 -lldec 0.0001 -hldec 0.0004 \
                                                -add_noop 2 --permute_hl 1 \
                                                -expl_wt 10 -expl_eta 4 \
                                                -col_coeff 0.0 \
                                                -motion 4 \
                                                -n_gpu 0 \
                                                -rollout 0 \
                                                -task 1 \
                                                -post -pre \
                                                -warm 100 \
                                                -neg_ratio 0. -opt_ratio 0.9 -dagger_ratio 0.1 \
                                                -ind 0 \
						-descr $TRIAL_NAME \
                                                -test namo_objs1_1/${TRIAL_NAME}_0 \
                                                -render

# pkill -f ros -9
# sleep 5


#done
#done

