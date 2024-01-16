import importlib
from opentamp.core.internal_repr import parameter
from opentamp.core.internal_repr.domain import Domain
from opentamp.core.internal_repr.parameter_schema import ParameterSchema
from opentamp.core.internal_repr.predicate_schema import PredicateSchema
from opentamp.core.internal_repr.action_schema import ActionSchema
from opentamp.errors_exceptions import DomainConfigException, PredicateException, ImpossibleException
import re

class ParseDomainConfig(object):
    """
    Read the domain configuration data and spawn the corresponding Domain object (see Domain class).
    """
    @staticmethod
    def parse(meta_config, action_config):

            return Domain(ParseDomainConfig._create_param_schemas(meta_config),
                          ParseDomainConfig._create_pred_schemas(meta_config),
                          ParseDomainConfig._create_action_schemas(action_config))

    @staticmethod
    def _create_param_schemas(meta_config):
        try:
            attr_imports = meta_config["attribute_imports"]
        except KeyError:
            attr_imports= {}
        
        attr_paths = {}
        for k in attr_imports:
            for v in attr_imports[k]:
                attr_paths[v] = importlib.import_module(k)

        # breakpoint()

        super_types = {}
        param_schemas = {}
        for t in meta_config['types']:
            super_types[t] = [t]
            param_schemas[t] = {"_type" : eval("str"), "name" : eval("str")} # name added by default

        # Subtypes will inherit all attributes of their parent, but can override
        parent_types = {}
        if "subtypes" in meta_config:
            for parent, children in meta_config['subtypes']:
                for child in children:
                    parent_types[child] = parent
                    param_schemas[child] = {"_type" : eval("str"), "name" : eval("str")}

        # First pass reads off explicitly defined values
        # for k, type_name, v in meta_config["object_attrs"]:
        #     param_schemas[type_name][k] = v
        #     if v in attr_paths:
        #         if not hasattr(attr_paths[v], v):
        #             raise DomainConfigException("%s not found in module %s!"%(v, attr_paths[v]))
        #         param_schemas[type_name][k] = getattr(attr_paths[v], v)
        #     else:
        #         try:
        #             param_schemas[type_name][k] = eval(v)
        #         except NameError as e:
        #             raise DomainConfigException("Need to provide attribute import path for non-primitive %s."%v)

        for type in meta_config['type_attrs']:
            for k, v in meta_config['type_attrs'][type].items():
                param_schemas[type][k] = v
                if v in attr_paths:
                    if not hasattr(attr_paths[v], v):
                        raise DomainConfigException("%s not found in module %s!"%(v, attr_paths[v]))
                    param_schemas[type][k] = getattr(attr_paths[v], v)
                else:
                    try:
                        param_schemas[type][k] = eval(v)
                    except NameError as e:
                        raise DomainConfigException("Need to provide attribute import path for non-primitive %s."%v)

        # Second pass performs inheritance
        def inherit(cur_type, base_type):
            if cur_type not in parent_types: return
            parent = parent_types[cur_type]
            parent_attrs = param_schemas[parent]
            for key in parent_attrs:
                if key == 'pose' and 'value' in param_schemas[base_type]: continue
                if key == 'value' and 'pose' in param_schemas[base_type]: continue
                if key not in param_schemas[base_type]:
                    param_schemas[base_type][key] = param_schemas[parent][key]
            if base_type not in super_types:
                super_types[base_type] = [base_type]

            super_types[base_type].append(parent)
            inherit(parent, base_type)

        for t in param_schemas: inherit(t, t)

        for type_name, attr_dict in list(param_schemas.items()):
            assert "pose" in attr_dict or "value" in attr_dict
            obj_or_symbol = ParseDomainConfig._dispatch_obj_or_symbol(attr_dict)  # for now, beliefs are objects
            param_schemas[type_name] = ParameterSchema(type_name, getattr(parameter, obj_or_symbol), attr_dict, super_types[type_name])

        return param_schemas

    @staticmethod
    def _create_pred_schemas(meta_config):
        try:
            pred_path = importlib.import_module(meta_config["predicates_import_path"])
        except KeyError as e:
            raise e

        pred_schemas = {}
        # for p_defn in domain_config["Derived Predicates"].split(";"):
        #     p_type, exp_types = map(str.strip, p_defn.split(",", 1))
        #     if not hasattr(common_predicates, p_type):
        #         raise PredicateException("Predicate type '%s' not defined!"%p_type)
        #     pred_schemas[p_type] = PredicateSchema(p_type, getattr(common_predicates, p_type), [s.strip() for s in exp_types.split(",")])
        for p_type, exp_types in meta_config["predicates"].items():
            if not hasattr(pred_path, p_type):
                raise PredicateException("Predicate type '%s' not defined!" % p_type)
            pred_schemas[p_type] = PredicateSchema(p_type, getattr(pred_path, p_type),
                                                   exp_types)

        return pred_schemas

    @staticmethod
    def _build_predicate_str(p_s):
        pred_strs = []
        count, prev_i = 0, 0
        for i, token in enumerate(p_s):
            if token == "(":
                count += 1
            if token == ")":
                count -= 1
                if count == 0:
                    next_str = p_s[prev_i:i+1].strip()
                    if next_str.find('(when') < 0:
                        pred_strs.append(p_s[prev_i:i+1].strip())
                    prev_i = i + 1
        return pred_strs

    @staticmethod
    def _create_action_schemas(action_config):
        action_schemas = {}
        for a_name, a_dict in list(action_config.items()):
            # # parse out params, predicates, time ranges
            # count, inds = 0, [0]
            # for i, token in enumerate(v):
            #     if token == "(":
            #         count += 1
            #     if token == ")":
            #         count -= 1
            #         if count == 0:
            #             inds.append(i+1)

            # params_str = v[inds[0]:inds[1]].strip()
            # pre = v[inds[1]:inds[2]].strip()
            # m = re.match("\(\s*and", pre)
            # if m:
            #     pre = pre[m.span()[1]:-1].strip()
            # eff = v[inds[2]:inds[3]].strip()
            # m = re.match("\(\s*and", eff)
            # if m:
            #     eff = eff[m.span()[1]:-1].strip()

            # pre_pred_strs = ParseDomainConfig._build_predicate_str(pre)
            # eff_pred_strs = ParseDomainConfig._build_predicate_str(eff)
            # pred_strs = pre_pred_strs + eff_pred_strs

            # all_active_timesteps = [tuple(map(int, s.split(":"))) for s in v[inds[-1]:].strip().split()]
            # # build list of params
            # params = []
            # for p in params_str.strip("()").split("?"):
            #     if p:
            #         p_name, p_type = p.strip().split("-")
            #         params.append(("?%s"%p_name.strip(), p_type.strip()))
            # # build universally quantified params
            # univ_params = {}
            # excl_params = {}
            # for i, pred in enumerate(pred_strs):
            #     while True:
            #         m = re.match("\(\s*forall", pred)
            #         if not m:
            #             break
            #         pred = pred[m.span()[1]:-1].strip()
            #         # print('Pred')
            #         # print(pred)
            #         g = re.match("\((.*?)\)(.*)", pred).groups()
            #         # print('g')
            #         # print(g)
            #         v = g[0].split("/")
            #         # print('v')
            #         # print(v)
            #         loop_var_name, loop_var_type = list(map(str.strip, v[0].split("-")))
            #         # print('Loop Var Name')
            #         # print(loop_var_name)
            #         # print('Loop Var Type')
            #         # print(loop_var_type)
            #         pred = g[1].strip()
            #         # if this dummy variable name is already used, then change the name
            #         unique_loop_var_name = loop_var_name
            #         ind = 1
            #         while unique_loop_var_name in univ_params:
            #             unique_loop_var_name = "{0}_{1}".format(loop_var_name, ind)
            #             ind += 1
            #         pred = pred.replace(loop_var_name, unique_loop_var_name)
            #         univ_params[unique_loop_var_name] = loop_var_type

            #         # The '/' operator denotes parameters to exclude from universal quant.
            #         excl_params[unique_loop_var_name] = [e.strip() for e in v[1:]]
            #         # replace this predicate in pred_strs because we removed the forall part
            #         # (and possibly renamed the dummy variable)
            #         pred_strs[i] = pred
            # # build preds
            # preds = []
            # for i, pred in enumerate(pred_strs):
            #     # handle not
            #     m = re.match("\(\s*not", pred)
            #     if m:
            #         pred = pred[m.span()[1]:-1].strip()
            #     negated = m is not None
            #     # parse out predicate type and args
            #     spl = pred.strip("() ").split()
            #     pred_type, args = spl[0], spl[1:]
            #     hl_info = None
            #     if i < len(pre_pred_strs):
            #         hl_info = "pre"
            #     else:
            #         hl_info = "eff"
            #     preds.append({"type": pred_type, "hl_info": hl_info, "args": args, "negated": negated,
            #                     "active_timesteps": all_active_timesteps[i], "ind": i})

            horizon = a_dict['horizon']
            
            # parse the predicates
            univ_params = {}
            excl_params = {}            
            predicates = []
            i = 0
            for precond in a_dict['pre']:
                precond_alt =  ParseDomainConfig._process_pred(precond, univ_params, excl_params, 'pre', i)
                predicates.append(precond_alt)
                i += 1
            
            for effect in a_dict['eff']:
                effect_alt = ParseDomainConfig._process_pred(effect, univ_params, excl_params, 'eff', i)
                predicates.append(effect_alt)
                i += 1
            
            non_deterministic = a_dict.get("non_deterministic", False)

            action_schemas[a_name] = ActionSchema(a_name, horizon, a_dict['params'], univ_params, predicates, excl_params, non_deterministic=non_deterministic)

        return action_schemas

    @staticmethod
    def _process_pred(pred, univ_params, excl_params, hl_info, idx):
        while 'forall' in pred:
            # universally quantified statement, create the new variable as universally quantified, enforce uniqueness, and give exceptions
            tmp_name = pred['forall'][0]
            while tmp_name in univ_params:
                tmp_name += '_' # add underscores to make names unique
            univ_params[tmp_name] = pred['forall'][1]
            if len(pred['forall']) > 2:
                excl_params[tmp_name] = pred['forall'][2:]
            else:
                excl_params[tmp_name] = {}
            pred = pred['pred']

        pred = ParseDomainConfig._parse_dict_pred(pred, hl_info, idx)
        return pred

    @staticmethod
    def _parse_dict_pred(pred, hl_info, idx):
        pred['hl_info'] = hl_info
        # ~ is a special character, denoting negation, in the "type" key
        pred['negated'] = (pred['type'][0] == '~')
        if pred['negated']:
            pred['type'] = pred['type'][1:] # remove ~ from type
        pred['active_timesteps'] = tuple(pred['active_timesteps'])
        pred['ind'] = idx
        return pred

    @staticmethod
    def _dispatch_obj_or_symbol(attr_dict):
        # decide whether this parameter is an Object or Symbol by looking at whether
        # it has an instance attribute named "pose" or one named "value" in the config file
        if "pose" in attr_dict:
            return "Object"
        elif "value" in attr_dict:
            return "Symbol"
        else:
            raise ImpossibleException("Can never reach here.")
