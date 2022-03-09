from core.internal_repr.state import State
from core.internal_repr.problem import Problem
from core.util_classes.learning import PostLearner
import copy
import random

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

    def is_hl_node(self):
        return False

    def is_ll_node(self):
        return False

    def plan(self):
        raise NotImplementedError("Override this.")

class HLSearchNode(SearchNode):
    def __init__(self, abs_prob, domain, concr_prob, priority=0, prefix=None, label='', llnode=None):
        self.abs_prob = abs_prob
        self.domain = domain
        self.concr_prob = concr_prob
        self.prefix = prefix if prefix is not None else []
        self.priority = priority
        self.label = label
        self.ref_plan = llnode.curr_plan if llnode is not None else None
        self.expansions = 0

    def is_hl_node(self):
        return True

    def plan(self, solver, debug=False):
        plan_obj = solver.solve(self.abs_prob, self.domain, self.concr_prob, self.prefix, label=self.label, debug=debug)
        if self.ref_plan is not None:
            if len(self.ref_plan.actions) < len(self.prefix):
                raise IndexError('ref_plan must be compatible with prefix')
            try:
                plan_obj.fill(self.ref_plan, amin=0, amax=len(self.prefix)-1)
            except AttributeError:
                raise ValueError('Task Planner failed to find a feasible plan')

            plan_obj.start = len(self.prefix)
            ts = (0, plan_obj.actions[plan_obj.start].active_timesteps[0])
            print('PREFIX SUCCESS:', plan_obj.get_failed_preds(active_ts=ts, tol=1e-3))
        return plan_obj

class LLSearchNode(SearchNode):
    def __init__(self, plan, prob, priority=1, keep_failed=False):
        self.curr_plan = plan
        self.concr_prob = prob
        self.child_record = {}
        self.priority = priority
        self._solved = None
        self.keep_failed = keep_failed # If true, replanning is done from the end of the first failed action instead of the start
        self.expansions = 0


    def parse_state(self, plan, failed_preds, ts, all_preds=[]):
        new_preds = [p for p in failed_preds if p is not None]
        reps = [p.get_rep() for p in new_preds]
        for a in plan.actions:
            a_st, a_et = a.active_timesteps
            if a_st > ts: break
            preds = copy.copy(a.preds)
            for p in all_preds:
                preds.append({'pred': p, 'active_timesteps':(0,0), 'hl_info':'hl_state', 'negated':False})
            for p in preds:
                if p['pred'].get_rep() in reps:
                    continue
                reps.append(p['pred'].get_rep())
                st, et = p['active_timesteps']
                if p['pred'].hl_include: 
                    new_preds.append(p['pred'])
                    continue
                if p['pred'].hl_ignore:
                    continue
                # Only check before the failed ts, previous actions fully checked while current only up to priority
                # TODO: How to handle negated?
                check_ts = ts - p['pred'].active_range[1]
                if st <= ts and check_ts >= 0 and et >= st:
                    # hl_state preds aren't tied to ll state
                    if p['pred'].hl_include:
                        new_preds.append(p['pred'])
                    elif p['hl_info'] == 'hl_state':
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
                            new_preds.append(p['pred'])
                        elif not p['negated'] and p['pred'].hl_test(0, tol=1e-3):
                            new_preds.append(p['pred'])

                        for param, attr in old_vals:
                            getattr(param, attr)[:,0] = old_vals[param, attr]
                    elif not p['negated'] and p['pred'].hl_test(check_ts, tol=1e-3):
                        new_preds.append(p['pred'])
                    elif p['negated'] and not p['pred'].hl_test(check_ts, tol=1e-3, negated=True):
                        new_preds.append(p['pred'])
        return new_preds


    def get_problem(self, i, failed_pred, failed_negated, suggester):
        """
        Returns a representation of the search problem which starts from the end state of step i and goes to the same goal.
        """
        state_name = "state_{}".format(self.priority)
        state_params = self.curr_plan.params.copy()
        if not len(self.curr_plan.actions) or self.curr_plan.actions[-1].active_timesteps[1] < i:
            print('BAD GET PROBLEM', failed_pred, i, self.curr_plan.actions)
            state_timestep = 0
            anum = 0
        else:
            anum, last_action = [(a_ind, a) for a_ind, a in enumerate(self.curr_plan.actions) if a.active_timesteps[0] <= i and a.active_timesteps[1] >= i][0]
            if self.keep_failed:
                anum += 1
                last_action = self.curr_plan.actions[anum]
            state_timestep = last_action.active_timesteps[0]
            # state_timestep = 0
        init_preds = self.curr_plan.prob.init_state.preds
        preds = []
        if failed_negated:
            preds = [failed_pred]
        state_preds = self.parse_state(self.curr_plan, preds, state_timestep, init_preds)
        state_preds.extend(self.curr_plan.hl_preds)
        # state_preds = [p['pred'] for p in last_action.preds if p['hl_info'] != 'eff' and  p['negated'] == False and p['active_timesteps'][0]==a.active_timesteps[0]]
        # for p in state_preds:
        #     arange = p.active_range
        #     p.active_range = (arange[0], max(arange[1], state_timestep))
        # state_preds.append(failed_pred)
        invariants = self.curr_plan.prob.init_state.invariants
        new_state = State(state_name, state_params, state_preds, state_timestep, invariants)
        """
        Suggester should sampling based on biased Distribution according to learned theta for each parameter.
        """
        if suggester != None:
            feature_fun = None
            resampled_action = suggester.sample(state, feature_fun)
        """
        End of Suggester
        """
        goal_preds = self.concr_prob.goal_preds.copy()
        new_problem = Problem(new_state, goal_preds, self.concr_prob.env, False, start_action=anum)
        return new_problem

    def solved(self):
        if self._solved is None:
            return len(self.curr_plan.get_failed_preds()) == 0
        return self._solved

    def is_ll_node(self):
        return True

    def gen_plan(self, hl_solver):
        self.curr_plan = hl_solver.get_plan(self.plan_str, self.domain, self.prob, self.initial)
        if type(self.curr_plan) is str: return
        if self.ref_plan is not None:
            self.curr_plan.fill(self.ref_plan, amax=len(self.ref_plan.actions)-1)

    def plan(self, solver, n_resamples=5, debug=False):
        self.curr_plan.freeze_actions(self.curr_plan.start)
        success = solver._backtrack_solve(self.curr_plan, anum=self.curr_plan.start, n_resamples=n_resamples, debug=debug)
        self._solved = success

    def get_failed_pred(self, forward_only=False):
        st = 0
        if forward_only:
            anum = self.curr_plan.start
            st = self.curr_plan.actions[anum].active_timesteps[0]
        et = self.curr_plan.horizon - 1
        failed_pred = self.curr_plan.get_failed_pred(active_ts=(st,et), hl_ignore=True)
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
