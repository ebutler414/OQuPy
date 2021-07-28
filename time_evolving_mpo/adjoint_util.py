from time_evolving_mpo.adjoint_method import SimpleAdjointTensor
from time_evolving_mpo.process_tensor import BaseProcessTensor, SimpleProcessTensor, FileProcessTensor

from typing import Text

def import_adjoint_process_tensor(
        filename: Text,
        process_tensor_type: Text = None) -> BaseProcessTensor:
    """
    ToDo.
    """
    pt_file = FileProcessTensor(mode="read", filename=filename)

    if process_tensor_type is None or process_tensor_type == "file":
        pt = pt_file
    elif process_tensor_type == "simple":
        pt = SimpleProcessTensor(
            hilbert_space_dimension=pt_file.hilbert_space_dimension,
            dt=pt_file.dt,
            transform_in=pt_file.transform_in,
            transform_out=pt_file.transform_out,
            name=pt_file.name,
            description=pt_file.description,
            description_dict=pt_file.description_dict)
        pt.set_initial_tensor(pt_file.get_initial_tensor())
    elif process_tensor_type == "adjoint":
        pt = SimpleAdjointTensor(
            hilbert_space_dimension=pt_file.hilbert_space_dimension,
            dt=pt_file.dt,
            transform_in=pt_file.transform_in,
            transform_out=pt_file.transform_out,
            name=pt_file.name,
            description=pt_file.description,
            description_dict=pt_file.description_dict)
        pt.set_initial_tensor(pt_file.get_initial_tensor())

        step = 0
        while True:
            try:
                mpo = pt_file.get_mpo_tensor(step, transformed=False)
            except IndexError:
                break
            pt.set_mpo_tensor(step, mpo)
            step += 1

        step = 0
        while True:
            cap = pt_file.get_cap_tensor(step)
            if cap is None:
                break
            pt.set_cap_tensor(step, cap)
            step += 1

        step = 0
        while True:
            lam = pt_file.get_lam_tensor(step)
            if lam is None:
                break
            pt.set_lam_tensor(step, lam)
            step += 1

    else:
        raise ValueError("Parameter 'process_tensor_type' must be "\
            + "'file' or 'simple'!")

    return pt

