import os
import numpy as np
import pandas as pd
from IPython.core.display import SVG
import pyomo.environ as pyo
import idaes
from idaes.core.solvers import use_idaes_solver_configuration_defaults
import idaes.core.util.scaling as iscale
import idaes.core.util as iutil
from idaes_examples.mod.power_gen import ngcc
import pytest
import logging
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
logging.getLogger("pyomo").setLevel(logging.ERROR)
m = pyo.ConcreteModel()
m.fs = ngcc.NgccFlowsheet(dynamic=False)
iscale.calculate_scaling_factors(m)
m.fs.initialize(
    load_from="ngcc_init.json.gz",
    save_to="ngcc_init.json.gz",
)
print(" degrees of freedom:", degrees_of_freedom(m))
use_idaes_solver_configuration_defaults()
idaes.cfg.ipopt.options.nlp_scaling_method = "user-scaling"
idaes.cfg.ipopt.options.linear_solver = "ma57"
idaes.cfg.ipopt.options.OF_ma57_automatic_scaling = "yes"
idaes.cfg.ipopt.options.ma57_pivtol = 1e-5
idaes.cfg.ipopt.options.ma57_pivtolmax = 0.1
solver = pyo.SolverFactory("ipopt")
res = solver.solve(m, tee=True)

m.fs.cap_specific_compression_power.fix(0)
m.fs.reboiler_duty_eqn.deactivate()
m.fs.st.reboiler.control_volume.properties_in[0].flow_mol.fix(0)

assert degrees_of_freedom(m) == 0
res = solver.solve(m, tee=True)