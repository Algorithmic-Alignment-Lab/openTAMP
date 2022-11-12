# openTAMP
OpenTAMP is an open-source library for optimization-based Task and Motion Planning (TAMP), and [Guided Imitation of TAMP](https://openreview.net/forum?id=-JwmfQC6IRt) with Python. OpenTAMP aims to make defining and solving new TAMP problems both easy and straightforward, even for users familiar with only the high-level ideas behind TAMP.

## Installation and Setup 


### Ubuntu
To install and begin using OpenTAMP on an Ubuntu (>14.04) Linux Machine, follow these steps:
1. Install Poetry by following instructions from [here](https://python-poetry.org/docs/#installation)
1. Make sure you have libglew (`sudo apt-get install libglfw3 libglew2.1`) and patchelf (`sudo apt-get install patchelf`) installed.
1. Install [MuJoCo](https://mujoco.org/)
    1. Make sure you install openmpi for linux (for use with MuJoCo): `sudo apt install libopenmpi-dev`
    1. Download the correct MuJoCo binary for your OS from [here](https://mujoco.org/download). Be sure to use version 2.1.0 and not a higher version!
    1. Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`
1. Clone the OpenTAMP repository from GitHub to a folder of your choice: `https://github.com/Algorithmic-Alignment-Lab/OpenTAMP.git`
1. `cd` into the newly-installed library and run `poetry shell`, then `poetry install`
1. Now, you should have a nice [virtual environment](https://realpython.com/python-virtual-environments-a-primer/) with python configured to run OpenTAMP! Whenever you want to use this, simply `cd` into the OpenTAMP folder and then run `poetry shell`
1. (Optional) If you'd like to use [Gurobi](https://www.gurobi.com/) as a backend solver for motion-planning problems, then follow steps [here](https://www.gurobi.com/wp-content/plugins/hd_documentations/content/pdf/quickstart_mac_8.1.pdf) to obtain and activate a license (note: free licenses are available for students and academic users!)
    1. Note that for obtaining a license, you must either install gurobi [via conda or from source](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-)

### Mac
To install and begin using OpenTAMP on a Mac, follow these steps
1. Install Poetry by following instructions from [here](https://python-poetry.org/docs/#installation)
1. Run ```brew install cmake glfw hdf5 ```
1. Install [MuJoCo](https://mujoco.org/)
    1. Download the correct MuJoCo binary for your OS from [here](https://mujoco.org/download). Be sure to use version 2.1.0 and not a higher version!
    1. Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`
    1. The first time you run the code, you may need to set the security permissions under settings to allow it to execute the mujoco binary
    1. Add this to you zshrc (or bashrc is using bash): `export DYLD_LIBRARY_PATH=$(brew --prefix)/lib:$DYLD_LIBRARY_PATH`
    1. You may encounter compiler errors on an Intel chip for the viewer, see [here](https://github.com/deepmind/dm_control/issues/276) for info (as of writing this, no solution was provided) 
1. Clone the OpenTAMP repository from GitHub to a folder of your choice: `https://github.com/Algorithmic-Alignment-Lab/OpenTAMP.git`
1. `cd` into the newly-installed library and run `poetry shell`, then `poetry install`
1. Setup Fast Downward (see [here](#fdsetup)). The provided binary Fast-Forward was compiled against Ubuntu, and unforunately Fast-Forward is written against an old standard of C that can no longer compile. 
1. Now, you should have a nice [virtual environment](https://realpython.com/python-virtual-environments-a-primer/) with python configured to run OpenTAMP! Whenever you want to use this, simply `cd` into the OpenTAMP folder and then run `poetry shell`
1. (Optional) If you'd like to use [Gurobi](https://www.gurobi.com/) as a backend solver for motion-planning problems, then follow steps [here](https://www.gurobi.com/wp-content/plugins/hd_documentations/content/pdf/quickstart_mac_8.1.pdf) to obtain and activate a license (note: free licenses are available for students and academic users!)
    1. Note that for obtaining a license, you must either install gurobi [via conda or from source](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-)
1. (Optional) If you want to use a Mac M1 gpu for training, run `pip install --force-reinstall --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu` and then go to `https://developer.apple.com/metal/pytorch/` for details on Metal

#### Troubleshooting
For some issues encountered on a Mac:
1. For an M1 chip, you may need to install x86 versions of various packages. Reference [here](https://medium.com/mkdir-awesome/how-to-install-x86-64-homebrew-packages-on-apple-m1-macbook-54ba295230f) for a guide on how to do so
1. If you see `ImportError: Failed to load GLFW3 shared library`, try running `pip install --force-reinstall glfw`
1. If you see `No module named 'importlib_metadata'` try running `pip install --force-reinstall gym`


### <a name="fdsetup"></a>Setup FastDownward Planner
The code provides wrappers to invoke the Fast-Forward  and the Fast Downward task planners. Fast-Forward  is provided through a pre-compiled binary and hence can be run without extra setup (but unfortunately only from Ubuntu). To setup Fast Downward, we need to take the following steps.
1. Move to the downward directory `cd opentamp/task_planners/downward`
1. Run: 
    ```
    git submodule init
    git submodule update
    ```
1. Build the downward binary by executing `./build.py`

### Verify planning

Try running `python opentamp/debug_scripts/test_grip_det.py` from the *root of the repository*, and if this script successfully completes and displays a short video at the end, your installation is correct!

This should take under a minute to run.


### Install Learning Dependencies
To run code related to learning, install these extra dependencies
1. `poetry install --extras learning`
1. If on a Mac, run `brew install python-tk`

### Verify learning (OUTDATED!!! WILL UPDATE SOON)
If you wish to train policies from the code, verify that a Mujoco key titled `mjkey.txt` is in your home directory an that `FULL_INSTALL=true` in `setup.sh`. Once this has completed, make sure you are on the virtual env (by running `tampenv`) and try running `test_training.sh`; this script will attempt to train policies for a two object pick-place problem with some default parameter settings. Once completed, is will generate some plots on performance and videos of rollouts into the `~/Dropbox` directory. Note the script will use about 16 proccesses, so it's reccomended to run it from at least an 8-physical core machine.


## Code Overview

### Defining Domains/Problems
One of the main goals of OpenTAMP is to make both designing and understanding TAMP problems straightforward. To this end, we take inspiration from  [PDDL](https://www.cs.toronto.edu/~sheila/2542/s14/A1/introtopddl2.pdf) and require both 'domain' and 'problem' files. At a high-level, domain files specify the types and dynamics of a particular problem domain (analogous to a class definition from object-oriented programming) and problem files specify the concrete objects and initial conditions of a particular instantiation of a domain (similar to an instance from object-oriented programming). In OpenTAMP, domain files are meant to provide an easy-to-understand but incomplete overview of a domain; details such as how particular types are implemented, or how particular constraints and their gradients are defined are defined elsewhere and must be imported.

At a high level, OpenTAMP domain definitions require:

- Types: all parameters must be one of these types. There is no specific code defining these that gets imported.
- Attribute Import Paths: these specify paths to Python classes that describe parameter attributes (which are used to define Primitive Predicates)
- Predicates Import Path: this specifies a path to Python classes defining all necessary predicates.
- Subtypes: these specify subtypes and their inheritance structure with the types defined above.
- Primitive Predicates: these specify continuous-valued object attributes (i.e, all objects of a certain type will have these attributes). Together, all the objects and their attributes define the low-level state of a problem (which TAMP will use motion planning to search through). NOTE that calling these 'predicates' is a slight abuse of terminology, since these are not boolean functions.
- Derived Predicates: these specify boolean-valued predicates defined over objects of certain types. Together, these predicates specify the high-level, abstract state of a problem (which TAMP will use task planning to search through).
- Actions: these specify the actions available to the agent. The action's preconditions and effects must be in terms of the derived predicates, and the definition must also specify timesteps over which the action itself, as well as its preconditions and effects, hold.

#### Domain files
Specifications for all domains should be placed in the `domains` folder directly under the `opentamp` directory

For a concrete example, refer to `opentamp/domains/namo_domain/generate_namo_domain.py`. This is script designed to generate a domain file (these files are cumbersome to write directly by hand).

##### Types
The first portion of the file will look like

`Types: Can, Target, RobotPose, Robot, Grasp, Obstacle`

Here, everything following `Types:` is a "type" that parameters in the domain can take on.

##### Import Paths
The next portion specifies where to find various necessary code

First:

`Attribute Import Paths: RedCircle opentamp.core.util_classes.items, Vector1d opentamp.core.util_classes.matrix, Vector2d opentamp.core.util_classes.matrix, Wall opentamp.core.util_classes.items, NAMO opentamp.core.util_classes.robots`

Here, we tell the planner where to find the python classes parameter attributes can take on (used in Primitive Predicates)

Second:

`Predicates Import Path: opentamp.core.util_classes.namo_predicates`

Here, we tell the planner where to find the python classes defining predicates

##### Listing predicates
The next portion specifies primitive and derived predicates

First:

`Primitive Predicates: geom, Can, RedCircle; pose, Can, Vector2d` etc...

This defines a list of 3-tuples of the form `(attribute_name, parameter_type, attribute_type)` and effectively tells the planner how to build parameters

Second:

`Derived Predicates: At, Can, Target; RobotAt, Robot, RobotPose; InGripper, Robot, Can, Grasp; Obstructs, Robot, Target, Target, Can`

This defines a list of tuples whose first element is a predicate class and the remaining are the type signatures for parameters of the predicate. In the above, the `At` predicate is defined to take both a `Can` object and a `Target` symbol

##### Defining actions
An action has five components: a name, a number of timesteps, a list of typed arguments, a "pre" list, and a "post" list
In the generate file above, these are defined as attributes of an `Action` class

Suppose we have a moveto action.

For args:
`args = '(?robot - Robot ?can - Can ?target - Target ?sp - RobotPose ?gp - RobotPose ?g - Grasp)'`

Means the action takes in a robot, a can, a target, a start pose, a goal pose, and a grasp

The `pre` list contains both precondition and midconditions for the action (originally, everything here was preconditions but that was impractical)

Preconditions are identified by being active at `0:0` i.e. only the first timestep

Items of this list are pairs of strings, the first describing the constraint and the second specifying what timesteps to enforce that constraint on

`('(At ?can ?target)', '0:0')` specifies that a precondition should be enforced constraining `?can` to be at `?target`

`('(forall (?w - Obstacle) (not (RCollides ?robot ?w)))', '0:{0}'.format(end))` specifies that for all objects of type `Obstacle` (not just those in the action arguments), the action should include a constraint prohibiting collision between the object and the robot from time 0 to time `end`

Note the syntax: `forall` allows ranging over the global space of parameters, not just those immediatley visible to the action, while `not` enforces the negation of a constraint (here, `RCollides` is a constraint to be in collision, so adding the `not` instead contrains to avoid collision)


The `eff` list is similar, but everything in this list will be considered a postcondition by the task planner and is something that must be true when the action finishes.

`('(forall (?obj - Can) (Stationary ?obj))', '{0}:{1}'.format(end, end-1))`

Note in the above how the timesteps are `(end, end-1)`; this is special syntax telling the motion planner to ignore the constraint even though the task planner will include it, and allows adding extra information to guide the task planner that is uncessary for the motion planner

#### Problem files
Specifications for all problem files should be placed under the relevant domain directory 

For a concrete example, refer to `domains/namo_domain/generate_namo_prob.py`. This is script designed to generate a problem file (these files are cumbersome to write directly by hand)

A problem file will define a list of objects, initial attribute values for those objects, a goal, and a set of initial conditions

The first part is of the form `Objects: ...`; everything on this line is of the form `ObjectType (name {insert name})` and will list EVERY object and symbol the planner will have access to. Semicolons delimit separate objects and the end of this line must be two new lines `\n\n`

The next part is of the form `Init: ...`; everything in this part is of the form `(attribute objectname value)` and specifies the initial value of every attribute for every object.

`(pose can0 [0,0])` for example specifies that the initial pose of `can0` will be at `[0,0]`

`(pose can1 undefined)` on the other hand specifies that the initial pose of `can1` is not fixed and the planner will determine its value. Items here are delimited by commas and the end of this part must be a semicolon.

The next part is then a list of predicates; these are constraints that ARE true at the initial timesteps. These must be satisfied by the initial values from the preceding part. This part is ended with two new lines `\n\n`

Finally, the last line is of the form `GOAL: ...` and specifies what constraints you want to be true at the end of the planning process.

During planning, the list of initial predicates will be converted to the initial state in PDDL while the goal will be converted likewise

#### Predicates
Predicates will define both constraints for the trajectory optimization problem as well as PDDL-style predicates for the task planner

All predicates will be subclasses of `ExprPredicate` from `core.util_classes.common_predicates`

There are roughly two types of predicates: linear and non-linear

Each predicate has a priority which can be interpreted as order-to-add: the solver will iteratively optimize problems with constraints restricted up-to the current priority, only adding the next priority when the current problem is solved. Usually, higher priority means a harder to solve constraint like collision avoidance

An example linear predicate: 
```
class At(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None):
        self.can, self.targ = params
        attr_inds = OrderedDict([(self.can, [("pose", np.array([0,1], dtype=np.int))]),
                                 (self.targ, [("value", np.array([0,1], dtype=np.int))])])

        A = np.c_[np.eye(2), -np.eye(2)]
        b = np.zeros((2, 1))
        val = np.zeros((2, 1))
        aff_e = AffExpr(A, b)
        e = EqExpr(aff_e, val)
        super(At, self).__init__(name, e, attr_inds, params, expected_param_types, priority=-2)
```
What's going on here:
- `attr_inds` is a dictionary describing which indices of which attributes will be used from each parameter; e.g. indices 0 and 1 of the can's pose will be included in the state vector x
- `A` defines a matrix such that Ax = [0., 0.] iff. x[:2] == x[2:]; in this context that means can.pose == targ.value
- `aff_e` is an affine expression of the form `Ax+b`
- `e` is then an affine constraint of the form `Ax+b=val`

Predicates can be defined over multiple timesteps

An example multi-timestep predicate:
```
class RobotStationary(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.c,  = params
        attr_inds = OrderedDict([(self.c, [("pose", np.array([0, 1], dtype=np.int))])])
        A = np.array([[1, 0, -1, 0],
                      [0, 1, 0, -1]])
        b = np.zeros((2, 1))
        e = EqExpr(AffExpr(A, b), np.zeros((2, 1)))
        super(RobotStationary, self).__init__(name, e, attr_inds, params, expected_param_types, active_range=(0,1), priority=-2)
```
What's going on here:
- `active_range` implies the state vector will be constructed using both the current timestep and the next; generally an `active_range` of the form `(a,b)` implies the state vector will use every timestep from `cur_ts+a` to `cur_ts+b`; using negative values to specifiy earlier timesteps


An example non-linear predicate
```
class InGraspAngle(ExprPredicate):
    def __init__(self, name, params, expected_param_types, env=None, sess=None, debug=False):
        self.r, self.can = params
        self.dist = 0.6
        attr_inds = OrderedDict([(self.r, [("pose", np.array([0, 1], dtype=np.int)),
                                           ("theta", np.array([0], dtype=np.int))]),
                                 (self.can, [("pose", np.array([0, 1], dtype=np.int))]),
                                ])

        def f(x):
            targ_loc = [-self.dist * np.sin(x[2]), self.dist * np.cos(x[2])]
            can_loc = x[3:5]
            return np.array([[((x[0]+targ_loc[0])-can_loc[0])**2 + ((x[1]+targ_loc[1])-can_loc[1])**2]])

        def grad(x):
            curdisp = x[3:5] - x[:2]
            theta = x[2]
            targ_disp = [-self.dist * np.sin(theta), self.dist * np.cos(theta)]
            off = (curdisp[0]-targ_disp[0])**2 + (curdisp[1]-targ_disp[1])**2
            (x1, y1), (x2, y2) = x[:2], x[3:5]

            x1_grad = -2 * ((x2-x1)+self.dist*np.sin(theta))
            y1_grad = -2 * ((y2-y1)-self.dist*np.cos(theta))
            theta_grad = 2 * dist * ((x2-x1)*np.cos(theta) + (y2-y1)*np.sin(theta))
            x2_grad = 2 * ((x2-x1)+self.dist*np.sin(theta))
            y2_grad = 2 * ((y2-y1)-self.dist*np.cos(theta))
            return np.array([x1_grad, y1_grad, theta_grad, x2_grad, y2_grad]).reshape((1,5))

        self.f = f
        self.grad = grad
        angle_expr = Expr(f, grad)
        e = EqExpr(angle_expr, np.zeros((1,1)))
        super(InGraspAngle, self).__init__(name, e, attr_inds, params, expected_param_types, priority=1)
```
What's going on here:
- This is a constraint specifiying that `can` must be at distance `0.6` along direction `theta` from `r`
- `f` replaces `aff_expr` from above with a black-box function call; `f` can only take the state vector as input and can only return a 1-d array
- `grad` will return the jacobian of `f` with respect to the state vector, or more precisely an array of the gradients of each output of `f`. `grad` can only return a 2-d array of shape `(len(f(x)), len(x))` such that `grad(x).dot(x) + f(x)` is valid
- `e` is an equality constraint of the form `f(x)=0`; `angle_expr` provides both `f` and it's gradient `grad` to the solver

## Debugging Tips
### Debugging Task Planning failures
If task planning is failing, it's a good idea to check the `temp/` folder under the main repository (note: this folder is only generated once you actually try to run task planning to solve a problem!). In particular, `temp/_temp_prob.output` will show the output of running [Fast Forward](https://planning.wiki/ref/planners/ff) on the problem specified, which can help catch subtle issues like if the goal is already achieved, or impossible, etc.

### Installing `sco-py` locally and placing breakpoints within it
It is sometimes useful to be able to place breakpoints within `sco-py` code to inspect and debug issues with motion planning problems. To do so, follow these steps:
1. Install `sco-py` locally `https://github.com/Algorithmic-Alignment-Lab/sco_py.git`
1. In the `pyproject.toml` file within the opentamp repo, replace the line `sco-py = { git = "https://github.com/Algorithmic-Alignment-Lab/sco.git", branch = "main" }` with `sco-py = { path = "<relative-path-to-sco>", develop=true }`
1. run `poetry update sco-py` and then `poetry install`

You can now place breakpoints within `sco-py` code! Be sure to revert `pyproject.toml` when you make a pull request.

