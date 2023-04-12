from .parameter_schema import ParameterSchema
from .predicate_schema import PredicateSchema
from .action_schema import ActionSchema

class Domain(object):
    """
    A single Domain object gets created every time p_mod_abs is run.
    It stores the parameter, predicate, and action schemas (see each of these classes).
    """
    def __init__(self, param_schemas, pred_schemas, action_schemas, belief=None):
        assert isinstance(param_schemas, dict)
        assert isinstance(pred_schemas, dict)
        assert isinstance(action_schemas, dict)
        for k, v in list(param_schemas.items()):
            assert isinstance(v, ParameterSchema)
        for k, v in list(pred_schemas.items()):
            assert isinstance(v, PredicateSchema)
        for k, v in list(action_schemas.items()):
            assert isinstance(v, ActionSchema)
        self.param_schemas = param_schemas
        self.pred_schemas = pred_schemas
        self.action_schemas = action_schemas