from queue import PriorityQueue

from core.internal_repr.plan import Plan
from core.parsing.parse_domain_config import ParseDomainConfig
from core.parsing.parse_problem_config import ParseProblemConfig
from core.parsing.parse_solvers_config import ParseSolversConfig
from core.util_classes.learning import PostLearner

from .prg_search_node import HLSearchNode, LLSearchNode


"""
Many methods called in p_mod_abs have detailed documentation.
"""


def p_mod_abs(
    hl_solver,
    ll_solver,
    domain,
    problem,
    initial=None,
    goal=None,
    suggester=None,
    max_iter=25,
    debug=False,
    label="",
    n_resamples=5,
    smoothing=False,
):
    if goal is None and problem.goal_test():
        return None, "Goal is already satisfied. No planning done."

    n0 = HLSearchNode(
        hl_solver.translate_problem(problem, initial, goal),
        domain,
        problem,
        priority=0,
        label=label,
    )

    Q = PriorityQueue()
    Q.put((n0.heuristic(), n0))
    for cur_iter in range(max_iter):
        if Q.empty():
            break
        n = Q.get_nowait()[1]
        if n.is_hl_node():
            c_plan = n.plan(hl_solver, debug)
            if c_plan == Plan.IMPOSSIBLE:
                print("IMPOSSIBLE PLAN IN PR GRAPH")
                if debug:
                    print("Found impossible plan")
                continue
            c = LLSearchNode(c_plan, n.concr_prob, priority=n.priority + 1)
            Q.put((n.heuristic(), n))
            Q.put((c.heuristic(), c))
        elif n.is_ll_node():
            n.plan(ll_solver, n_resamples=n_resamples, debug=debug)
            if n.solved():
                print("SOLVED PR GRAPH")
                if smoothing:
                    suc = ll_solver.traj_smoother(n.curr_plan, n_resamples=n_resamples)
                return n.curr_plan, None
            Q.put((n.heuristic(), n))
            if n.gen_child():
                # Expand the node
                fail_step, fail_pred, fail_negated = n.get_failed_pred()
                n_problem = n.get_problem(fail_step, fail_pred, fail_negated, suggester)
                abs_prob = hl_solver.translate_problem(n_problem, goal=goal)
                prefix = n.curr_plan.prefix(fail_step)
                c = HLSearchNode(
                    abs_prob,
                    domain,
                    n_problem,
                    priority=n.priority + 1,
                    prefix=prefix,
                    label=label,
                    llnode=n,
                )
                Q.put((c.heuristic(), c))

        if debug:
            if n.is_hl_node():
                print(
                    "Current Iteration: HL Search Node with priority {}".format(
                        n.priority
                    )
                )
            elif n.is_ll_node():
                print(
                    "Current Iteration: LL Search Node with priority {}".format(
                        n.priority
                    )
                )
                print(f"plan str: {n.curr_plan.get_plan_str()}")

        import ipdb; ipdb.set_trace()

    print(("PR GRAPH hit iteration limit {0}".format(label)))
    return None, "Hit iteration limit, aborting."
