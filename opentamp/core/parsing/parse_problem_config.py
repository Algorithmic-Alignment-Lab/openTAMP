import os

import pybullet as P
from opentamp.core.util_classes.viewer import PyBulletViewer

from opentamp.core.internal_repr import problem, state
from opentamp.errors_exceptions import ProblemConfigException

try:
    import tensorflow as tf
except:
    pass


class ParseProblemConfig(object):
    """
    Read the problem configuration data and spawn the corresponding initial Problem object (see Problem class).
    This is only done for spawning the very first Problem object, from the initial state specified in the problem configuration file.
    Validation is performed against the schemas stored in the Domain object self.domain.
    """

    @staticmethod
    def parse(
        problem_config,
        domain,
        env=None,
        openrave_bodies={},
        reuse_params=None,
        initial=None,
        visual=False,
        use_tf=False,
        sess=None,
    ):
        # create parameter objects
        # params = {}

        # if "objects" not in problem_config or not problem_config["objects"]:
        #     raise ProblemConfigException("Problem file needs objects.")
        # for t in problem_config["objects"]:
        #     o_type, attrs = list(map(str.strip, t.strip(" )").split("(", 1)))
        #     attr_dict = {}
        #     for l in map(str.strip, attrs.split(".")):
        #         lst = l.split(" ", 1)
        #         k, v = lst[0], lst[1:]
        #         attr_dict[k] = v
        #     assert "name" in attr_dict
        #     name = attr_dict["name"][0]
        #     attr_dict["_type"] = [o_type]
        #     params[name] = attr_dict

        # if "init" not in problem_config or not problem_config["init"]:
        #     raise ProblemConfigException("Problem file needs init.")
        # prim_preds, deriv_preds = list(
        #     map(str.strip, problem_config["Init"].split(";"))
        # )
        # if prim_preds:
        #     for pred in map(str.strip, prim_preds.split(")")):
        #         if pred:
        #             a, b = pred.find("["), pred.rfind("]") + 1
        #             if a != -1:
        #                 new_s = "".join(pred[a:b].split())
        #                 pred = pred.replace(pred[a:b], new_s)
        #             lst = list(map(str.strip, pred.strip(",() ").split()))
        #             k = lst[0]
        #             obj_name = lst[1]
        #             v = lst[2:]
        #             if obj_name not in params:
        #                 raise ProblemConfigException(
        #                     "'%s' is not an object in problem file." % obj_name
        #                 )
        #             params[obj_name][k] = [
        #                 x.replace("[", "(").replace("]", ")") for x in v
        #             ]
        #     if reuse_params is not None:
        #         params = reuse_params
        #     else:
        #         for obj_name, attr_dict in list(params.items()):
        #             o_type = attr_dict["_type"][0]
        #             name = attr_dict["name"][0]
        #             try:
        #                 params[obj_name] = domain.param_schemas[o_type].param_class(
        #                     attrs=attr_dict,
        #                     attr_types=domain.param_schemas[o_type].attr_dict,
        #                     class_types=domain.param_schemas[o_type].types,
        #                 )
        #                 if obj_name in openrave_bodies:
        #                     params[obj_name].openrave_body = openrave_bodies[obj_name]
        #                     params[obj_name].geom = params[obj_name].openrave_body._geom
        #             except KeyError as e:
        #                 raise ProblemConfigException(
        #                     "Parameter '%s' not defined in domain file." % name
        #                 )
        #             except ValueError as e:
        #                 print(e)
        #                 raise ProblemConfigException(
        #                     "Some attribute type in parameter '%s' is incorrect." % name
        #                 )
        # for k, v in list(params.items()):
        #     if type(v) is dict:
        #         raise ProblemConfigException(
        #             "Problem file has no primitive predicates for object '%s'." % k
        #         )
        # init_preds = set()
        # if initial is not None:
        #     for i, pred in enumerate(initial):
        #         spl = list(map(str.strip, pred.strip("() ").split()))
        #         p_name, p_args = spl[0], spl[1:]
        #         p_objs = []
        #         for n in p_args:
        #             try:
        #                 p_objs.append(params[n])
        #             except KeyError:
        #                 raise ProblemConfigException(
        #                     "Parameter '%s' for predicate type '%s' not defined in domain file."
        #                     % (n, p_name)
        #                 )

        #         init_preds.add(
        #             domain.pred_schemas[p_name].pred_class(
        #                 name="initpred%d" % i,
        #                 params=p_objs,
        #                 expected_param_types=domain.pred_schemas[
        #                     p_name
        #                 ].expected_params,
        #                 env=env,
        #             )
        #         )

        # elif deriv_preds:
        #     for i, pred in enumerate(deriv_preds.split(",")):
        #         spl = list(map(str.strip, pred.strip("() ").split()))
        #         if not len(spl):
        #             continue
        #         p_name, p_args = spl[0], spl[1:]
        #         p_objs = []
        #         for n in p_args:
        #             try:
        #                 p_objs.append(params[n])
        #             except KeyError:
        #                 raise ProblemConfigException(
        #                     "Parameter '%s' for predicate type '%s' not defined in domain file."
        #                     % (n, p_name)
        #                 )
        #         init_preds.add(
        #             domain.pred_schemas[p_name].pred_class(
        #                 name="initpred%d" % i,
        #                 params=p_objs,
        #                 expected_param_types=domain.pred_schemas[
        #                     p_name
        #                 ].expected_params,
        #                 env=env,
        #             )
        #         )
        #         # except Exception as e:
        #         #    print(e)
        #         #    import ipdb; ipdb.set_trace()

        # # Invariant predicates are enforced every timestep
        # invariant_preds = problem_config.get("Invariants", None)
        # invariant_set = set()
        # if invariant_preds:
        #     for i, pred in enumerate(invariant_preds.split(",")):
        #         spl = list(map(str.strip, pred.strip("() ").split()))
        #         if not len(spl):
        #             continue
        #         p_name, p_args = spl[0], spl[1:]
        #         p_objs = []
        #         for n in p_args:
        #             try:
        #                 p_objs.append(params[n])
        #             except KeyError:
        #                 raise ProblemConfigException(
        #                     "Parameter '%s' for predicate type '%s' not defined in domain file."
        #                     % (n, p_name)
        #                 )
        #         invar_pred = domain.pred_schemas[p_name].pred_class(name="invariantpred%d"%i,
        #                                                                   params=p_objs,
        #                                                                   expected_param_types=domain.pred_schemas[p_name].expected_params,
        #                                                                   env=env)
        #         invariant_set.add(invar_pred)
        #         init_preds.add(invar_pred)


        # # create goal predicate objects
        # goal_preds = set()
        # for i, pred in enumerate(problem_config["Goal"].split(",")):
        #     spl = list(map(str.strip, pred.strip("() ").split()))
        #     p_name, p_args = spl[0], spl[1:]
        #     p_objs = []
        #     for n in p_args:
        #         try:
        #             p_objs.append(params[n])
        #         except KeyError:
        #             raise ProblemConfigException(
        #                 "Parameter '%s' for predicate type '%s' not defined in domain file."
        #                 % (n, p_name)
        #             )
        #     goal_preds.add(
        #         domain.pred_schemas[p_name].pred_class(
        #             name="goalpred%d" % i,
        #             params=p_objs,
        #             expected_param_types=domain.pred_schemas[p_name].expected_params,
        #             env=env,
        #         )
        #     )

        if env is None:
            try:
                P.disconnect()
            except:
                pass

            if not visual:
                env = P.connect(P.DIRECT)
                P.resetSimulation()
            else:
                pbv = PyBulletViewer()
                pbv = pbv.create_viewer()
                env = pbv.env

        params = {}
        init_preds = set()
        invariant_set = set()
        
        if reuse_params is not None:
            params = reuse_params
        else:
            for obj_init in problem_config['init_objs']:
                obj_type = obj_init['type']
                obj_init['_type'] = obj_type
                del obj_init['type']
                obj_name = obj_init['name']
                try:
                    params[obj_name] = domain.param_schemas[obj_type].param_class(
                        attrs=obj_init,
                        attr_types=domain.param_schemas[obj_type].attr_dict,
                        class_types=domain.param_schemas[obj_type].types,
                    )
                    if obj_name in openrave_bodies:
                        params[obj_name].openrave_body = openrave_bodies[obj_name]
                        params[obj_name].geom = params[obj_name].openrave_body._geom
                except KeyError as e:
                    raise ProblemConfigException(
                        "Parameter '%s' not defined in domain file." % obj_name
                    )
                except ValueError as e:
                    print(e)
                    raise ProblemConfigException(
                        "Some attribute type in parameter '%s' is incorrect." % obj_name
                    )

        for i, init_pred in enumerate(problem_config['init_preds']):
            p_objs = []
            for param_name in init_pred['args']:
                try:
                    p_objs.append(params[param_name])
                except KeyError:
                    raise ProblemConfigException(
                        "Parameter '%s' for predicate type '%s' not defined in domain file."
                        % (param_name, init_pred['type'])
                    )
            init_preds.add(
                domain.pred_schemas[init_pred['type']].pred_class(
                    name="initpred%d" % i,
                    params=p_objs,
                    expected_param_types=domain.pred_schemas[
                        init_pred['type']
                    ].expected_params,
                    env=env,
                )
            )
            
        if 'invariants' in problem_config:
            for i, invariant in enumerate(problem_config['invariants']):
                p_objs = []
                for param_name in invariant['args']:
                    try:
                        p_objs.append(params[param_name])
                    except KeyError:
                        raise ProblemConfigException(
                            "Parameter '%s' for predicate type '%s' not defined in domain file."
                            % (param_name, invariant['type'])
                        )
                
                invar_pred = domain.pred_schemas[invariant['name']].pred_class(
                        name="invariantpredpred%d" % i,
                        params=p_objs,
                        expected_param_types=domain.pred_schemas[
                            invariant['type']
                        ].expected_params,
                        env=env,
                    )
                
                invariant_set.add(invar_pred)
                init_pred.add(invar_pred)

        goal_preds = set()
        for i, goal_p in enumerate(problem_config['goals']):
            p_objs = []
            
            for param_name in goal_p['args']:
                try:
                    p_objs.append(params[param_name])
                except KeyError:
                    raise ProblemConfigException(
                        "Parameter '%s' for predicate type '%s' not defined in domain file."
                        % (param_name, goal_p['type'])
                    )
            
            goal_preds.add(
                domain.pred_schemas[goal_p['type']].pred_class(
                    name="goalpred%d" % i,
                    params=p_objs,
                    expected_param_types=domain.pred_schemas[goal_p['type']].expected_params,
                    env=env,
                )
            )
        
        # use params and initial preds to create an initial State object
        initial_state = state.State(
            "initstate", params, init_preds, timestep=0, invariants=invariant_set
        )


        # use initial state to create Problem object
        initial_problem = problem.Problem(initial_state, goal_preds, env, sess=sess)
        return initial_problem
