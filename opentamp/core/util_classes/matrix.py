import numpy as np

class Matrix(np.ndarray):
    """
    The matrix class is useful for tracking object poses.
    """
    def __new__(cls, *args, **kwargs):
        raise NotImplementedError("Override this.")

class Vector(Matrix):
    """
    The vector class.
    """
    def __new__(cls, vec):
        if type(vec) is str:
            if not vec.endswith(")"):
                vec += ")"
            vec = eval(vec)
        obj = np.array(vec, dtype=np.float32)
        # deals with case where obj is zero-dimensional
        assert len(np.atleast_1d(obj)) == cls.dim
        obj = obj.reshape((cls.dim, obj.size//cls.dim))
        return obj

class Vector1d(Vector):
    dim = 1

class Vector2d(Vector):
    """
    The NAMO domain uses the Vector2d class to track poses of objects in the grid.
    """
    dim = 2

class Vector3d(Vector):
    dim = 3

class Vector5d(Vector):
    dim = 5

class Vector7d(Vector):
    dim = 7

class PR2PoseVector(Vector3d):
    """
    The PR2 domain uses the PR2PoseVector class to base poses of pr2.
    """
    pass

class PR2ArmPose(Vector7d):
    """
    The PR2's arm pose is a 7d vector.
    """
    pass

class Value(Vector1d):
    pass

class ArmPose5d(Vector5d):
    """
    The 5 dimensional arm pose is a 5d vector.
    """
    pass

class ArmPose7d(Vector7d):
    """
    The 7 dimensional arm pose is a 7d vector.
    """
    pass

class DimVectorZeroed(Vector):
    def __new__(cls, vec):
        if type(vec) is str:
            if not vec.endswith(")"):
                vec += ")"
            vec = eval(vec)
        obj = np.zeros(shape=(vec.item(),), dtype=np.float32)
        # deals with case where obj is zero-dimensional
        assert len(np.atleast_1d(obj)) == cls.dim
        obj = obj.reshape((cls.dim, obj.size//cls.dim))
        return obj