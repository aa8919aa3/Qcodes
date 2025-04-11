import qcodes as qc
from qcodes.dataset.measurements import Measurement
from qcodes.parameters import ManualParameter

# Create a parameter to measure
param_to_measure = ManualParameter("measured_param", unit="V")

# Create a parameter to sweep
param_to_sweep = ManualParameter("sweep_param", unit="A")

# Create an experiment (for organizing datasets)
exp = qc.Experiment(name="my_experiment", sample_name="my_sample")

# Perform a measurement
with Measurement(exp=exp, name="my_measurement") as m:
    m.register_parameter(param_to_sweep, setpoints=(param_to_sweep,))
    m.register_parameter(param_to_measure, setpoints=(param_to_sweep,))

    for sweep_value in [0.1, 0.2, 0.3]:
        param_to_sweep.set(sweep_value)
        param_to_measure.set(sweep_value**2)  # Example measurement
        m.add_result((param_to_sweep, sweep_value), (param_to_measure, param_to_measure.get()))

    dataset = m.run()

print(f"Data saved to: {dataset.location}")