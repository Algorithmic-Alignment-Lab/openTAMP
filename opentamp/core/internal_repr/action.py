MAX_PRIORITY = 3
class Action(object):
    """
    An instantiated action stores the following.

    step_num: step number of this action in the plan
    name: name of this action
    active_timesteps: (start_time, end_time) for this action
    params: ordered list of Parameter objects
    preds: list of dictionaries where each dictionary contains information about
    a predicate. each dictionary contains
    - the Predicate object
    - negated (Boolean)
    - hl_info (string) which is "pre", "post" and "hl_state" if the predicate is
      a precondition, postcondition, or neither and part of the high level state
      respectively
    - active_timesteps (tuple of (start_time, end_time))
    """
    def __init__(self, step_num, name, active_timesteps, params, preds, train_policy=False, linked_action=None, non_deterministic=False):
        self.step_num = step_num
        self.name = name
        self.active_timesteps = active_timesteps
        self.params = params
        self.preds = preds
        self.train_policy = train_policy
        self.linked_action = linked_action
        self.non_deterministic = non_deterministic
        # make effect attainment in all nondeterministic actions true
        if self.non_deterministic:
            for pred in self.preds:
                pred['pred'].optimistic = True

    def __repr__(self):
        return "%d: %s %s %s"%(self.step_num, self.name, self.active_timesteps, " ".join([p.name for p in self.params]))

    def get_failed_preds(self, active_ts=None, priority=MAX_PRIORITY, tol=1e-3, incl_negated=True):
        if active_ts is None:
            active_ts = self.active_timesteps
        failed = []
        for pred_d in self.preds:
            if pred_d['hl_info'] == 'hl_state': continue
            pred = pred_d['pred']
            negated = pred_d['negated']
            start, end = pred_d['active_timesteps']
            for t in range(max(start, active_ts[0]),
                           min(end, active_ts[1])+1):
                if pred.active_range[1]+t > active_ts[1] or pred.active_range[0] + t < active_ts[0]:
                    continue
                if (incl_negated or not negated) and pred.priority <= priority and not pred.test(t, negated=negated, tol=tol):
                    failed.append((negated, pred, t))
        return failed

    def get_failed_preds_by_type(self, active_ts=None, priority=MAX_PRIORITY, tol=1e-3):
        if active_ts is None:
            active_ts = self.active_timesteps
        failed = []
        for pred_d in self.preds:
            if pred_d['hl_info'] == 'hl_state': continue
            pred = pred_d['pred']
            negated = pred_d['negated']
            start, end = pred_d['active_timesteps']
            for t in range(max(start, active_ts[0]),
                           min(end, active_ts[1])+1):
                if pred.active_range[1]+t > active_ts[1] or pred.active_range[0] + t < active_ts[0]:
                    continue
                if pred.priority <= priority and not pred.test(t, negated=negated, tol=tol):
                    failed.append((t, pred.get_type(), pred.check_pred_violation(t, negated=negated, tol=tol).flatten()))

        return failed

    def get_pr_preds(self, ts, priority):
        res = []
        for pred_d in self.preds:
            if pred_d['hl_info'] == 'hl_state': continue
            start, end = pred_d['active_timesteps']
            if start >= ts[0] and end <= ts[1] and pred_d['pred'].priority == priority:
                res.append(pred_d['pred'])
        return res

    def get_active_preds(self, t):
        res = []
        for pred_d in self.preds:
            if pred_d['hl_info'] == 'hl_state': continue
            start, end = pred_d['active_timesteps']
            if start <= t and end >= t: res.append(pred_d['pred'])
        return res

    def get_all_active_preds(self):
        preds = set()
        ts_range = self.active_timesteps
        for t in range(ts_range[0], ts_range[1]+1):
            for res in self.get_active_preds(t):
                preds.add(res)
        return preds

    def satisfied(self, active_ts=None):
        if active_ts is None:
            active_ts = self.active_timesteps
        if self.active_timesteps[0] > active_ts[1] \
            or self.active_timesteps[1] < active_ts[0]:
            return True
        return len(self.get_failed_preds(active_ts)) == 0

    def first_failed_ts(self, priority = MAX_PRIORITY):
        start, end = self.active_timesteps
        ## init at the maximize
        t_min = end
        for b, p, t in self.get_failed_preds(priority=priority):
            if t < t_min:
                t_min = t
        return t_min
