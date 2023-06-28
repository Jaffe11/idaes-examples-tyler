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
from IPython.display import display
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
#printing out the unit variables
for v in m.fs.st.reboiler.component_data_objects(pyo.Var):
    print(v.name, v.value)



run_series = True
if run_series:
    idaes.cfg.ipopt.options.tol = 1e-6
    idaes.cfg.ipopt.options.max_iter = 50
    solver = pyo.SolverFactory("ipopt")

    m.fs.cap_specific_reboiler_duty.fix(2.4e6)
    m.fs.cap_fraction.fix(0.97)
    powers = np.linspace(650, 185, int((650 - 185) / 5) + 1)
    powers = list(powers)
    powers.insert(1, 646)

    df = pd.DataFrame(columns=m.fs.tags_output.table_heading())
    df1 = pd.DataFrame(index = powers, columns = ["HPTurb_steamflow","IPTurb_steamflow","LPTurb_steamflow","Ng_flow" ,
                                                  "FlueGas_lpEcon_flow", "FlueGas_lpEcon_temp","FlueGas_lpEcon_N2",
                                                  "FlueGas_lpEcon_O2", "FlueGas_lpEcon_CO2", "FlueGas_lpEcon_H2O",
                                                  "Steam_Electrical_poweroutput", "Ng_Electrical_poweroutput",
                                                  ])
    components=["N2", "O2", "CO2", "H2O"]
    for p in powers:
        print("Simulation for net power = ", p)
        fname = f"data/ngcc_{int(p)}.json.gz"
        if os.path.exists(fname):
            iutil.from_json(m, fname=fname, wts=iutil.StoreSpec(suffix=False))
        else:
            m.fs.net_power_mw.fix(p)
            res = solver.solve(m, tee=False, symbolic_solver_labels=True)
            if not pyo.check_optimal_termination(res):
                break
            iutil.to_json(m, fname=fname)
        df.loc[m.fs.tags_output["net_power"].value] = m.fs.tags_output.table_row(
            numeric=True
        )
        #Collecting data for surrogate modeling
        df1.loc[p,"HPTurb_steamflow"] = m.fs.st.steam_turbine.hp_stages[1].inlet.flow_mol[0].value # mol/s
        df1.loc[p,"IPTurb_steamflow"] = m.fs.st.steam_turbine.ip_stages[1].inlet.flow_mol[0].value # mol/s
        df1.loc[p,"LPTurb_steamflow"] = m.fs.st.steam_turbine.lp_stages[1].inlet.flow_mol[0].value # mol/s
        df1.loc[p,"Ng_flow"] = m.fs.tags_output["fuel_flow"].value #kg/s
        df1.loc[p, "Steam_Electrical_poweroutput"] = m.fs.tags_output["st_power"].value # MW
        df1.loc[p,"Ng_Electrical_poweroutput"] = m.fs.tags_output["gt_power"].value #MW
        df1.loc[p,"FlueGas_lpEcon_flow"] =sum( m.fs.hrsg.econ_lp.hot_side.properties_out[0].flow_mol_comp[c].value 
                                              for c in components) # mol/s
        df1.loc[p,"FlueGas_lpEcon_temp"] = m.fs.hrsg.econ_lp.hot_side.properties_out[0].temperature.value # K
        for c in components:
            df1.loc[p,"FlueGas_lpEcon_" +str(c)] =(m.fs.hrsg.econ_lp.hot_side.properties_out[0].flow_mol_comp[c].value /
                                                   sum( m.fs.hrsg.econ_lp.hot_side.properties_out[0].flow_mol_comp[c].
                                                       value for c in components))  # mol_comp

        if abs(p - 650) < 0.1:
            m.fs.gt.streams_dataframe().to_csv(
                "data_tabulated/ngcc_stream_650mw_gt.csv"
            )
            m.fs.st.steam_streams_dataframe().to_csv(
                "data_tabulated/ngcc_stream_650mw_st.csv"
            )
            m.fs.hrsg.steam_streams_dataframe().to_csv(
                "data_tabulated/ngcc_stream_650mw_hrsg_steam.csv"
            )
            m.fs.hrsg.flue_gas_streams_dataframe().to_csv(
                "data_tabulated/ngcc_stream_650mw_hrsg_gas.csv"
            )
    df.to_csv("data_tabulated/ngcc.csv")
    df1.to_csv("data_tabulated/surrogate_data.csv")

    # Display the results from the run stored in a pandas dataframe
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    display(df1)