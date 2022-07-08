from opentamp.errors_exceptions import DomainConfigException
from opentamp.core.util_classes.matrix import Vector
from opentamp.core.util_classes.openrave_body import OpenRAVEBody

import numpy as np


class Parameter(object):
    """
    Parameters fall into one of two categories: Objects and Symbols. Objects are
    things that exist in the environment with some pose, and Symbols are
    symbolic references. To store information about the environment that's
    useful for planning, we often spawn a workspace Object (but this is not
    necessary). Objects and Symbols are distinguished in the config file parsing
    as follows: Objects have a "pose" instance attribute while Symbols have a
    "value" instance attribute.
    """
    def __init__(self, attrs=None, attr_types=None, class_types=[]):
        self.openrave_body = None
        self._free_attrs = {}
        self._saved_free_attrs = {}
        self.attrs = []

        if attr_types is not None:
            self._attr_types = attr_types.copy()
            self._attr_types['_attr_types'] = dict
        else:
            self._attr_types = {'attr_types': dict}

        if attrs is not None:
            for attr_name, arg in list(attrs.items()):
                self.attrs.append(attr_name)
                if "undefined" in arg:
                    setattr(self, attr_name, "undefined")
                else:
                    try:
                        setattr(self, attr_name, attr_types[attr_name](*arg))
                    except KeyError:
                        name = attrs["name"][0]
                        raise DomainConfigException("Attribute '{}' for {} '{}' not defined in domain file.".format(attr_name, type(self).__name__, name))
            self.class_types = class_types if len(class_types) else [self._type]
            self.class_types = list(set(self.class_types))

    # def get_attr_type(self, attr_name):
    #     raise NotImplementedError("get_attr_type not implemented for Parameter.")

    def get_attr_type(self, attr_name):
        if attr_name == 'openrave_body':
            return OpenRAVEBody
        elif attr_name == '_free_attrs':
            return dict
        elif attr_name == '_saved_free_attrs':
            return dict
        elif attr_name == 'attrs':
            return list
        elif attr_name == 'class_types':
            return list

        try:
            attr = self._attr_types[attr_name]
        except Exception as e:
            print((self, self.name))
            raise e
        return attr

    def get_type(self, find_all=False):
        if find_all:
            types = self.class_types
            if hasattr(self, 'geom'):
                types.extend(self.geom.get_types())
            return list(set(types))

        return self._type

    def is_symbol(self):
        return False

    def is_fixed(self, attr_list, t=None):
        if t is None:
            return not np.all([np.all(self._free_attrs[attr]) for attr in attr_list])
        else:
            return not np.all([self._free_attrs[attr][:, t] for attr in attr_list])

    def is_defined(self):
        for attr_name in self._attr_types.keys():
            if getattr(self, attr_name) is "undefined":
                return False
        return True

    def save_free_attrs(self):
        self._saved_free_attrs = {}
        for k, v in list(self._free_attrs.items()):
            self._saved_free_attrs[k] = v.copy()

    def restore_free_attrs(self):
        self._free_attrs = self._saved_free_attrs

    def get_free_attrs(self):
        free_attrs = {}
        for k, v in list(self._free_attrs.items()):
            free_attrs[k] = v.copy()
        return free_attrs

    def store_free_attrs(self, free_attrs):
        self._free_attrs = free_attrs

    def freeze_up_to(self, t):
        if t <= 0: return
        for attr in self._free_attrs:
            self._free_attrs[attr][:,:t+1] = 0.

    def fix_attr(self, attr, active_ts):
        if self.is_symbol():
            active_ts = (0,0)
        self._free_attrs[attr][:,active_ts[0]:active_ts[1]+1] = 0

    def free_attr(self, attr, active_ts):
        if self.is_symbol():
            active_ts = (0,0)
        self._free_attrs[attr][:,active_ts[0]:active_ts[1]+1] = 1

    def fix_all_attr(self, active_ts):
        if self.is_symbol():
            active_ts = (0,0)
        for attr in self._free_attrs:
            self._free_attrs[attr][:,active_ts[0]:active_ts[1]+1] = 0

    def free_all_attr(self, active_ts):
        if self.is_symbol():
            active_ts = (0,0)
        for attr in self._free_attrs:
            self._free_attrs[attr][:,active_ts[0]:active_ts[1]+1] = 1

    def fill(self, param, active_ts):
        if self.is_symbol(): active_ts = (0, 0)
        st, et = active_ts
        for attr_name, v in list(self.__dict__.items()):
            attr_type = self.get_attr_type(attr_name)
            if issubclass(attr_type, Vector):
                getattr(self, attr_name)[:,st:et+1] = getattr(param, attr_name)[:,st:et+1]

    def set_to_time(self, ts):
        if hasattr(self, 'openrave_body') and self.openrave_body is not None:
            self.openrave_body.set_from_param(self, ts)

    def __repr__(self):
        return "%s - %s"%(self.name, self.get_type())

class Object(Parameter):
    """
    Objects must have at minimum a name, a type (a string), and a pose attribute
    (a d-by-T table, which we refer to as the trajectory table), which is set to
    "undefined" if the object's pose is not defined. The attributes for the
    objects are defined in the configuration files. attrs is a dictionary from
    instance attribute name to the argument to pass into the __init__ method for
    the class stored in the corresponding entry of attr_types. attrs must have,
    at minimum, the keys "name", "_type", and "pose".
    """
    def __init__(self, attrs=None, attr_types=None, class_types=[]):
        if attrs is not None:
            assert "name" in attrs and "_type" in attrs and "pose" in attrs
        super(Object, self).__init__(attrs=attrs, attr_types=attr_types, class_types=class_types)

    def is_defined(self):
        return self.pose is not "undefined"

    def copy(self, new_horizon, reset_free=False):
        new = Object()
        new_free = {}
        for attr_name, v in list(self.__dict__.items()):
            attr_type = self.get_attr_type(attr_name)
            if issubclass(attr_type, Vector):
                new_value = np.empty((attr_type.dim, new_horizon))
                new_value[:] = np.NaN
                if v is not "undefined":
                    assert attr_type.dim == v.shape[0]
                    new_value[:v.shape[0], :v.shape[1]] = v[:v.shape[0], :min(v.shape[1], new_horizon)].copy()
                setattr(new, attr_name, new_value)
                new_free[attr_name] = np.ones(new_value.shape)
                new_free[attr_name][:,0] = 0
            else:
                setattr(new, attr_name, v)
        if reset_free:
            new._free_attrs = new_free
        return new


class Symbol(Parameter):
    """
    Symbols must have at minimum a name, a type (a string), and a value
    attribute, which is set to "undefined" if the symbol's value is not defined.
    Symbols are static, which means that it's value does not depend on time.
    The attributes for the symbols are defined in the configuration files. attrs
    is a dictionary from instance attribute name to the arguments to pass into
    the __init__ method for the class stored in the corresponding entry of
    attr_types. attrs must have, at minimum, the keys "name", "_type", and
    "value".
    """
    def __init__(self, attrs=None, attr_types=None, class_types=[]):
        if attrs is not None:
            assert "name" in attrs and "_type" in attrs and "value" in attrs
        try:
            super(Symbol, self).__init__(attrs=attrs, attr_types=attr_types, class_types=class_types)
        except AssertionError:
            import ipdb; ipdb.set_trace()

    def is_defined(self):
        return self.value is not "undefined"

    def is_symbol(self):
        return True

    def copy(self, new_horizon, reset_free=False):
        new = Symbol()
        new_free = {}
        for k, v in list(self.__dict__.items()):
            if v == 'undefined':
                attr_type = self.get_attr_type(k)
                assert issubclass(attr_type, Vector)
                val = np.empty((attr_type.dim, 1))
                val[:] = np.NaN
                setattr(new, k, val)
                new_free[k] = np.ones(val.shape)
            elif hasattr(v, 'copy'):
                setattr(new, k, v.copy())
                if type(v) is np.ndarray:
                    new_free[k] = np.zeros_like(v) # self._free_attrs[k].copy()
            else:
                setattr(new, k, v)
        # if reset_free:
        #     new._free_attrs = new_free
        return new
