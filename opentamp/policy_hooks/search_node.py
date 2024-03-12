from opentamp.core.internal_repr.state import State
from opentamp.core.internal_repr.problem import Problem
from opentamp.core.util_classes.learning import PostLearner
import copy
import functools
import random
import numpy as np

DEBUG = False

@functools.total_ordering
class SearchNode(object):
    """
    There are two types of nodes in the plan refinement graph (PRGraph). High-level search
    nodes store abstract and concrete representations of the problem (concrete is an instance
    of the Problem class), and they interface to planning with the chosen HLSolver. Low-level search
    nodes store the Plan object for refinement, and they interface to planning with the chosen LLSolver.
    """
    def __init__(self, *args):
        raise NotImplementedError("Must instantiate either HL or LL search node.")

    def heuristic(self):
        """
        The node with the highest heuristic value is selected at each iteration of p_mod_abs.
        """
        return -self.priority

    def __lt__(self, node):
        self.heuristic() < node.heuristic()

    def is_hl_node(self):
        return False

    def is_ll_node(self):
        return False

    def plan(self):
        raise NotImplementedError("Override this.")

class HLSearchNode(SearchNode):
    def __init__(self, abs_prob, domain, concr_prob, priority=0, prefix=None, label='', llnode=None, x0=None, targets=None, expansions=0, nodetype='optimal', info={}, replan_start=None, conditioned_obs={}, observation_model=None, belief_true={}, path=[]):
        self.abs_prob = abs_prob
        self.domain = domain
        self.concr_prob = concr_prob
        self.prefix = prefix if prefix is not None else []
        self.priority = priority
        self.label = label
        self.ref_plan = llnode.curr_plan if llnode is not None else None
        self.targets = targets
        self.x0 = x0
        self.expansions = expansions
        self.llnode = llnode
        self.nodetype = nodetype
        self._trace = [label] 
        self.info = info
        if llnode is not None:
            self._trace.extend(llnode._trace)
        self.replan_start = replan_start
        self.conditioned_obs = conditioned_obs
        self.observation_model = observation_model
        self.belief_true = belief_true
        self.path = path

    # def is_hl_node(self):
    #     return True

    # def plan(self, solver):
    #     plan_obj = solver.solve(self.abs_prob, self.domain, self.concr_prob, self.prefix, label=self.label)
    #     if self.ref_plan is not None and type(plan_obj) is not str:
    #         if len(self.ref_plan.actions) < len(self.prefix):
    #             raise IndexError('ref_plan must be compatible with prefix')
    #         plan_obj.fill(self.ref_plan, amin=0, amax=len(self.prefix)-1)
    #         plan_obj.start = len(self.prefix)
    #         ts = (0, plan_obj.actions[plan_obj.start].active_timesteps[0])
    #         if DEBUG: print('PREFIX SUCCESS:', plan_obj.get_failed_preds(active_ts=ts, tol=1e-3))
    #     return plan_obj

class LLSearchNode(SearchNode):
    def __init__(self, plan_str, domain, prob, initial, priority=1, keep_failed=False, ref_plan=None, x0=None, targets=None, expansions=0, label='', refnode=None, freeze_ts=-1, hl=True, ref_traj=[], nodetype='optimal', env_state=None, info={}, replan_start=None, conditioned_obs={}, observation_model=None, belief_true={}, path=[]):
        self.curr_plan = 'no plan'
        self.plan_str = plan_str
        self.domain = domain
        self.initial = initial
        self.concr_prob = prob
        self.child_record = {}
        self.priority = priority
        self._solved = None
        self.targets = targets
        self.ref_plan = ref_plan
        self.x0 = x0
        self.env_state = env_state
        self.hl = hl
        self.freeze_ts = freeze_ts
        self.label = label
        self.ref_traj = ref_traj
        self.nodetype = nodetype
        self.info = info
        #self.refnode = refnode
        self._trace = [label] 
        if refnode is not None:
            self._trace.extend(refnode._trace)

        # If true, replanning is done from the end of the first failed action instead of the start
        # This is useful if trajectories are rolled out online and you do not wish to perform state resets
        self.keep_failed = keep_failed
        self.expansions = expansions
        self.replan_start = replan_start
        self.conditioned_obs = conditioned_obs
        self.observation_model = observation_model
        self.belief_true = belief_true
        self.path = path


    def parse_state(self, plan, failed_preds, ts, all_preds=[]):
        # build predicate array from the failed_preds buffer (including reps)
        new_preds = [p for p in failed_preds if p is not None]
        # reps = [p.get_rep() for p in new_preds]
        # in each action
        for a in plan.actions:
            # break if action occurs after the given ts
            a_st, a_et = a.active_timesteps
            if a_st > ts: break
            # construct a preliminary buffer of predicates (preds) which exist in this action
            preds_all = copy.copy(a.preds)
            preds = []
            for p in preds_all:
                if a.active_timesteps[0]<=p['active_timesteps'][0] and p['active_timesteps'][1]<=a.active_timesteps[1]:
                    preds.append(p)
            # for p in all_preds:
            #     preds.append({'pred': p, 'active_timesteps':(0,0), 'hl_info':'hl_state', 'negated':False})
            for p in preds:
                # if the given predicate is in the new_preds buffer already
                # if p['pred'].get_rep() in reps:
                #     ## if already contained, but is contained in a negated way, remove it
                
                # FILTER OUT NEGATION TODO TEST

                check_ts = ts - p['pred'].active_range[1]

                if check_ts >= 0: 
                    if p['negated'] and p['pred'].hl_test(check_ts, tol=1e-3, negated=True) and \
                            p['pred'].get_type() in [q['pred'].get_type() for q in new_preds if not q['negated']]:
                            for q in new_preds:
                                if q['pred'].get_type() == p['pred'].get_type():
                                    new_preds.remove(q)

                    if not p['negated'] and p['pred'].hl_test(check_ts, tol=1e-3, negated=False) and \
                            p['pred'].get_type() in [q['pred'].get_type() for q in new_preds if q['negated']]:
                            for q in new_preds:
                                if q['pred'].get_type() == p['pred'].get_type():
                                    new_preds.remove(q)
                #                     # reps.remove(p['pred'].get_rep())
                #     else:
                #         continue
                # add given pred to reps
                # reps.append(p['pred'].get_rep())
                st, et = p['active_timesteps']
                # include if hl_include without exception
                if p['pred'].hl_include: 
                    new_preds.append(p['pred'])
                    continue
                if p['pred'].hl_ignore:
                    continue
                # Only check before the failed ts, previous actions fully checked while current only up to priority
                # if predicate occurs in the applicable range
                if st <= ts and check_ts >= 0 and et >= st:
                    # NOTE hl_state preds aren't tied to ll state!
                    if p['pred'].hl_include:
                        # totally redundant logic as far as I can tell
                        new_preds.append(p)
                    elif p['hl_info'] == 'hl_state':
                        # add given predicate to the inital planned state if i
                        if p['pred'].active_range[1] > 0: continue
                        old_vals = {}
                        for param in p['pred'].attr_inds:
                            for attr, _ in p['pred'].attr_inds[param]:
                                if param.is_symbol():
                                    aval = getattr(plan.params[param.name], attr)[:,0]
                                else:
                                    aval = getattr(plan.params[param.name], attr)[:,check_ts]
                                old_vals[param, attr] = getattr(param, attr)[:,0].copy()
                                getattr(param, attr)[:,0] = aval
                        if p['negated'] and not p['pred'].hl_test(0, tol=1e-3, negated=True):
                            new_preds.append(p)
                        elif not p['negated'] and p['pred'].hl_test(0, tol=1e-3):
                            new_preds.append(p)

                        for param, attr in old_vals:
                            getattr(param, attr)[:,0] = old_vals[param, attr]
                    elif not p['negated'] and p['pred'].hl_test(check_ts, tol=1e-3):
                        new_preds.append(p)
                    elif p['negated'] and not p['pred'].hl_test(check_ts, tol=1e-3, negated=True):
                        new_preds.append(p)

        return [new_pred['pred'] for new_pred in new_preds]


    def get_problem(self, i, failed_pred, failed_negated, suggester=None):
        """
        Returns a representation of the search problem which starts from the end state of step i and goes to the same goal.
        """
        state_name = "state_{}".format(self.priority)
        state_params = self.curr_plan.params.copy()
        if not len(self.curr_plan.actions) or self.curr_plan.actions[-1].active_timesteps[1] < i:
            state_timestep = 0
            anum = 0
        else:
            anum, last_action = [(a_ind, a) for a_ind, a in enumerate(self.curr_plan.actions) if a.active_timesteps[0] <= i and a.active_timesteps[1] >= i][0]
            if self.keep_failed or last_action.non_deterministic:
                anum += 1
                last_action = self.curr_plan.actions[anum]
            state_timestep = last_action.active_timesteps[0]
        self.curr_plan.start = anum
        init_preds = self.curr_plan.prob.init_state.preds
        invariants = self.curr_plan.prob.init_state.invariants
        preds = []
        if failed_negated:
            preds = [failed_pred]
        state_preds = self.parse_state(self.curr_plan, preds, state_timestep, [])
        #breakpoint()
        state_preds.extend(self.curr_plan.hl_preds)
        new_state = State(state_name, state_params, state_preds, state_timestep, [])
        goal_preds = self.concr_prob.goal_preds.copy()
        new_problem = Problem(new_state, goal_preds, self.concr_prob.env, False, start_action=anum)

        return new_problem

    def solved(self):
        if self._solved is None:
            return len(self.curr_plan.get_failed_preds(tol=1e-3)) == 0
        return self._solved

    def is_ll_node(self):
        return True

    def gen_plan(self, hl_solver, bodies, ll_solver):
        self.curr_plan = hl_solver.get_plan(self.plan_str, self.domain, self.concr_prob, self.initial)

        # breakpoint()

        if type(self.curr_plan) is str: return
        if not len(self.curr_plan.actions):
            print('Search node found bad plan for', self.initial, self.plan_str, self.concr_prob.goal)

        if self.ref_plan is not None:
            self.curr_plan.start = self.ref_plan.start
            fill_a = self.ref_plan.start
            if fill_a >= 0:
                self.curr_plan.fill(self.ref_plan, amax=fill_a)


            for param in self.curr_plan.params.values():
                if not param.is_symbol(): continue
                for attr in param._free_attrs:
                    ref_sym = getattr(self.ref_plan.params[param.name], attr)
                    getattr(param, attr)[:] = ref_sym[:]


            if self.freeze_ts > 0:
                self.curr_plan.freeze_ts = self.freeze_ts
                #preds = self.curr_plan.get_failed_preds()
                #ref_preds = self.ref_plan.get_failed_preds()
                #if len(preds) and any([p[2] <= self.freeze_ts for p in preds]):
                #    if DEBUG: print('LLNODE: Violation in constraints, projecting onto', self._trace, preds, ref_preds, self.freeze_ts, self._trace, self.curr_plan.actions)
                #    try:
                #        proj_succ = ll_solver.find_closest_feasible(self.curr_plan, (0, self.freeze_ts))
                #    except Exception as e:
                #        if DEBUG:
                #            print('LLNODE FAIL:', e)
                #            print('LLNODE: Failed to project onto constraints', preds)
                #        proj_succ = False

                #preds = self.curr_plan.get_failed_preds()
                #preds = [p for p in preds if p[2]+p[1].active_range[1] <= self.freeze_ts]
                #if len(preds):
                #    if DEBUG: print('LLNODE: Proceeding without projection', self._trace, preds, self.freeze_ts)
                #else:
                #    if DEBUG: print('LLNODE: Proceeding with frozen', self._trace)
                #    self.curr_plan.freeze_up_to(self.freeze_ts)
                self.curr_plan.freeze_up_to(self.freeze_ts)

        return self.curr_plan

    def plan(self, solver, n_resamples=5):
        self.curr_plan.freeze_actions(self.curr_plan.start)
        success = solver._backtrack_solve(self.curr_plan, anum=self.curr_plan.start, n_resamples=n_resamples)
        self._solved = success

    def get_failed_pred(self, forward_only=False, st=0):
        if forward_only:
            anum = self.curr_plan.start
            st = self.curr_plan.actions[anum].active_timesteps[0]
        et = self.curr_plan.horizon - 1
        failed_pred = self.curr_plan.get_failed_pred(active_ts=(st,et), hl_ignore=True, tol=5e-2)
        if hasattr(failed_pred[1], 'hl_ignore') and failed_pred[1].hl_ignore:
            return failed_pred[2], None, failed_pred[0]
        return failed_pred[2], failed_pred[1], failed_pred[0]

    def gen_child(self):
        """
            Make sure plan refinement graph doesn't generate doplicate child.
            self.child_record is a dict that maps planing prefix to failed predicates it
            encountered so far.
        """

        fail_step, fail_pred, fail_negated = self.get_failed_pred()
        if fail_pred is None:
            return False

        plan_prefix = tuple(self.curr_plan.prefix(fail_step))
        fail_pred_type = fail_pred.get_type()

        if plan_prefix not in list(self.child_record.keys()) or fail_pred_type not in self.child_record.get(plan_prefix, []):
            self.child_record[plan_prefix] = self.child_record.get(plan_prefix, []) + [fail_pred_type]
            return True
        else:
            return False
