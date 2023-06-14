class ActionSchema(object):
    """
    An action schema holds the following information.

    name: name of this action
    horizon: number of allotted timesteps
    params: ordered list of (name, type) of "true" parameters
    universally_quantified_params: mapping of {name: type} of dummy params used in forall expressions
    preds: list containing, for each predicate,
    - type of this predicate
    - arguments
    - negated (Boolean)
    - active_timesteps (tuple of (start_time, end_time))
    """
    def __init__(self, name, horizon, params, universally_quantified_params, preds, exclude_params={}):
        self.name = name
        self.horizon = horizon
        self.params = params
        self.universally_quantified_params = universally_quantified_params
        self.exclude_params = exclude_params
        self.preds = preds
