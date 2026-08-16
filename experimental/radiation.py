import xtrack as xt

line: xt.Line = xt.load("test")
line.configure_bend_model(core="bend-kick-bend", edge="full")
line.configure_drift_model(model="exact")
tt: xt.Table = line.get_table()
tt_quad = tt.rows[tt.element_type == "Quadrupole"]
line.set(tt_quad, model="mat-kick-mat", integrator="yoshida4", num_multipole_kicks=7)

# Slice the line
slicing_strategies = [
    xt.Strategy(slicing=None),  # Default catch-all
    xt.Strategy(slicing=xt.Uniform(slicing_order=10, mode="thick"), element_type=xt.Bend),
    # xt.Strategy(slicing=xt.Uniform(slicing_order=10, mode='thick'), element_type=xt.Quadrupole),
]
line.slice_thick_elements(slicing_strategies)
