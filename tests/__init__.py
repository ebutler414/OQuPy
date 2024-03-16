import oqupy
import tensornetwork as tn
import numpy as np



correlations = oqupy.PowerLawSD(alpha=0.05,
                                zeta=1.0,
                                cutoff=5.0,
                                cutoff_type="exponential",
                                temperature=0.2,
                                name="ohmic")
correlations2 = oqupy.PowerLawSD(alpha=0.1,
                                zeta=1.0,
                                cutoff=5.0,
                                cutoff_type="exponential",
                                temperature=0.2,
                                name="ohmic")
bath = oqupy.Bath(0.5*oqupy.operators.sigma("z"),
                    correlations,
                    name="phonon bath")
bath2 = oqupy.Bath(0.5*oqupy.operators.sigma("z"),
                    correlations2,
                    name="half-coupling phonon bath")
tempo_params = oqupy.TempoParameters(dt=0.1,
                                     dkmax=5,
                                     epsrel=10**(-6))
pt = oqupy.pt_tempo_compute(bath,
                            start_time=0.0,
                            end_time=1.0,
                            parameters=tempo_params)
pt2 = oqupy.pt_tempo_compute(bath2,
                            start_time=0.0,
                            end_time=1.0,
                            parameters=tempo_params)

system = oqupy.System(oqupy.operators.sigma("x"))
initial_state = oqupy.operators.spin_dm("z+")

dyns = oqupy.compute_dynamics(system,
                                  process_tensor=[pt,pt],
                                  initial_state=initial_state)
