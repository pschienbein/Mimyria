
import os
from pathlib import Path
import shutil

from mimyria.autotrain import submit_job, is_job_running_in_path, get_job_exit_code_in_path
from mimyria.postprocess import calculate_apt_from_field, calculate_pgt_from_field
from mimyria.io import apt_flatten, pgt_flatten

import mimyria.cp2k_input_parser as cp2k

from ase.io import iread


class Backend:
    def atomic_polar_tensor(self, state, path, configs, **kwargs):
        raise NotImplementedError

    def polarizability_gradient_tensor(self, state, path, configs, **kwargs):
        raise NotImplementedError


class CP2kBackend(Backend):
    def apt_check_force_eval(self, fn = 'force_eval.inc'):
        with open(fn) as fin:
            cp2k_input = fin.read()

            # check if the section is present
            sec = 'FORCE_EVAL/DFT/PERIODIC_EFIELD'
            if not cp2k.is_section_present(cp2k_input, sec):
                raise RuntimeError(f"checking force_eval.inc: A periodic Efield section must be present: {sec}")

            # check if the important settings are variables
            ok, details = cp2k.check_variables_as_placeholders(
                    cp2k_input,
                    "FORCE_EVAL/DFT/PERIODIC_EFIELD",
                    {"INTENSITY": "FIELD_INTENSITY", "POLARISATION": "FIELD_POLARISATION"}
                    )

            if not ok:
                raise RuntimeError(f"checking force_eval.inc: The periodic Efield section is incorrectly configured: {details[0]['problems']}")

            # get the WFN restart
            restart = cp2k.get_keyword_value(cp2k_input, 'FORCE_EVAL/DFT', 'WFN_RESTART_FILE_NAME')
            if restart != 'field_mx-RESTART.wfn':
                print('[NOTE:] WFN_RESTART_FILE_NAME: Calculation might benefit from setting it to "field_mx-RESTART.wfn"')


    def atomic_polar_tensor(self, state, path, configs, **kwargs):
        # status flags
        bJobStarted = os.path.exists(path + '/job.id')
        bRunning = is_job_running_in_path(path)

        nconf = len(configs)

        # job not running and NOT finished
        if not bRunning and not bJobStarted:
            # check if the user provided force_eval.inc has the required settings
            # The function throws on error and might print some hints
            self.apt_check_force_eval()

            # copy the cp2k tamplate over, if it doesn't exist ye
            src = Path(__file__).resolve().parent.parent / 'templates' / 'cp2k' / 'apt-from-field'
            dst = Path(path)

            shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)

            # NOTE may require some more testing if everything went smoothly
            # NOTE This is waiting until the job completes
            submit_job(state, path, 'cpu', array=[1, nconf])

        # job running, but not finished
        # this implies that the job is either still running or an error occurred
        elif bRunning:
            raise RuntimeError('Generating reference data -- process still running or an error occurred while calculating, exiting....')

        # From here, this is either a new call to atomic_polar_tensor
        # and the job is finished
        # OR this is call waited for the job to complete 
        # -> else is not suitable
        retcodes = get_job_exit_code_in_path(path)

        files = {'px': 'field_px-frc-1.xyz',
                 'py': 'field_py-frc-1.xyz',
                 'pz': 'field_pz-frc-1.xyz',
                 'mx': 'field_mx-frc-1.xyz',
                 'my': 'field_my-frc-1.xyz',
                 'mz': 'field_mz-frc-1.xyz',
                 'pos': '../configurations.xyz'}
        frames2write = []

        # check all the return codes
        nfailed = 0
        for i in retcodes:
            if retcodes[i] != 0:
                print(f'Calculation of snapshot {i} failed, return code {retcodes[i]}:')
                self.print_error(retcodes[i])
                nfailed += 1

            else:
                res_dir = path + f'/{i:05d}'
                iterators = {key: iread(res_dir + '/' + fn) for key, fn in files.items()}

                for step, frames in enumerate(zip(*iterators.values())):
                    atoms = dict(zip(iterators.keys(), frames))

                    apts = calculate_apt_from_field(atoms, 5e-4)
                    print(apts.shape)

                    trajectory = atoms['pos']
                    # to store the data in the xyz file
                    # NOTE: this is row-major!
                    trajectory.arrays['apt'] = apt_flatten(apts)
                    frames2write.append(trajectory)

        if nfailed > 0:
            raise RuntimeError(f'Generating reference data: {nfailed}/{nconf} snapshots failed!')

        return frames2write


    ###############################################################################################
    # Polarizability Gradient Tensor
    def pgt_check_force_eval(self, fn = 'force_eval.inc'):
        with open(fn) as fin:
            cp2k_input = fin.read()

            # check if the section is present
            sec = 'FORCE_EVAL/DFT/PERIODIC_EFIELD'
            if not cp2k.is_section_present(cp2k_input, sec):
                raise RuntimeError(f"checking force_eval.inc: A periodic Efield section must be present: {sec}")

            # check if the important settings are variables
            ok, details = cp2k.check_variables_as_placeholders(
                    cp2k_input, sec,
                    {"INTENSITY": "FIELD_INTENSITY", "POLARISATION": "FIELD_POLARISATION"}
                    )

            if not ok:
                raise RuntimeError(f"checking force_eval.inc: The periodic Efield section is incorrectly configured: {details[0]['problems']}")

            # For the PGT, the field section must be guarded!
            ok, details = cp2k.check_section_guarded_by_if(cp2k_input, sec, "${FIELD_INTENSITY-0}")
            if not ok:
                raise RuntimeError(
                    f"checking force_eval.inc: {sec} must be guarded by "
                    f"@IF ${'{FIELD_INTENSITY-0}'} ... @ENDIF. Problems: {details[0]['problems']}"
                )

            # get the EPS_SCF level
            eps_scf = cp2k.get_keyword_value(cp2k_input, 'FORCE_EVAL/DFT/SCF', 'EPS_SCF')
            if eps_scf is None or float(eps_scf) >= 1e-8:
                print('[WARNING:] EPS_SCF is >= 1e-8. The set up calculation might be too inaccurate!')

            eps_scf_o = cp2k.get_keyword_value(cp2k_input, 'FORCE_EVAL/DFT/SCF/OUTER_SCF', 'EPS_SCF')
            if eps_scf_o is None or abs(float(eps_scf) - float(eps_scf_o)) > 1e-15:
                raise RuntimeError('checking force_eval.inc: EPS_SCF in OUTER_SCF does not agree with EPS_SCF in SCF')

            # get the WFN restart
            restart = cp2k.get_keyword_value(cp2k_input, 'FORCE_EVAL/DFT', 'WFN_RESTART_FILE_NAME')
            if restart != 'field_0-RESTART.wfn':
                print('[NOTE:] WFN_RESTART_FILE_NAME: Calculation might benefit from setting it to "field_0-RESTART.wfn"')


    def polarizability_gradient_tensor(self, state, path, configs, **kwargs):
        # status flags
        bJobStarted = os.path.exists(path + '/job.id')
        bRunning = is_job_running_in_path(path)

        nconf = len(configs)

        # job not running and NOT finished
        if not bRunning and not bJobStarted:
            # check if the user provided force_eval.inc has the required settings
            # The function throws on error and might print some hints
            self.pgt_check_force_eval()

            # copy the cp2k tamplate over, if it doesn't exist ye
            src = Path(__file__).resolve().parent.parent / 'templates' / 'cp2k' / 'pgt-from-field'
            dst = Path(path)

            shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)

            # NOTE may require some more testing if everything went smoothly
            # NOTE This is waiting until the job completes
            submit_job(state, path, 'cpu', array=[1, nconf])

        # job running, but not finished
        # this implies that the job is either still running or an error occurred
        elif bRunning:
            raise RuntimeError('Generating reference data -- process still running or an error occurred while calculating, exiting....')

        # check if calculation exited successfully
        retcodes = get_job_exit_code_in_path(path)

        # check the energy files, if enough frames have been calculated
        files = {'0': 'field_0-frc-1.xyz',
                 'px': 'field_px-frc-1.xyz',
                 'py': 'field_py-frc-1.xyz',
                 'pz': 'field_pz-frc-1.xyz',
                 'mx': 'field_mx-frc-1.xyz',
                 'my': 'field_my-frc-1.xyz',
                 'mz': 'field_mz-frc-1.xyz',
                 'pxpy': 'field_pxpy-frc-1.xyz',
                 'pxpz': 'field_pxpz-frc-1.xyz',
                 'pypz': 'field_pypz-frc-1.xyz',
                 'mxmy': 'field_mxmy-frc-1.xyz',
                 'mxmz': 'field_mxmz-frc-1.xyz',
                 'mymz': 'field_mymz-frc-1.xyz',
                 'pos': '../configurations.xyz'}

        frames2write = []
        
        # go through all calculations
        nfailed = 0
        for i in retcodes:
            if retcodes[i] != 0:
                print(f'Calculation of snapshot {i} failed, return code {retcodes[i]}:')
                self.print_error(retcodes[i])
                nfailed += 1

            else:
                res_dir = path + f'/{i:05d}'
                iterators = {key: iread(res_dir + '/' + fn) for key, fn in files.items()}

                for step, frames in enumerate(zip(*iterators.values())):
                    atoms = dict(zip(iterators.keys(), frames))

                    pgts = calculate_pgt_from_field(atoms, 5e-4)
                    print(pgts.shape)

                    trajectory = atoms['pos']
                    # to store the data in the xyz file
                    # NOTE: this is row-major!
                    trajectory.arrays['pgt'] = pgt_flatten(pgts)
                    # trajectory.arrays['pgt'] = pgts.reshape(len(pgts), 27)
                    frames2write.append(trajectory)

        if nfailed > 0:
            self.print_error(retcode)
            raise RuntimeError(f'Generating reference data: {nfailed}/{nconf} snapshots failed!')

        return frames2write


    def print_error(self, retcode):
        if retcode == 0:
            print('No Error')
        elif retcode == 1:
            print('Execution of CP2k failed')
        elif retcode == 2:
            print('SCF did not converge')
        else:
            print(f'Unknown Error Code {retcode}')


# Just for more abstraction
class APTCalculator:
    def __init__(self, calc):
        self.calc = calc

    def calculate(self, state, path, configs, **kwargs):
        return self.calc.atomic_polar_tensor(state, path, configs, **kwargs)


class PGTCalculator:
    def __init__(self, calc):
        self.calc = calc

    def calculate(self, state, path, configs, **kwargs):
        return self.calc.polarizability_gradient_tensor(state, path, configs, **kwargs)


def get_calculator(state):
    calculators = {
            'apt': APTCalculator,
            'pgt': PGTCalculator
            }
    backends = {
            'cp2k': CP2kBackend
            }

    return calculators[state['target']](backends[state['backend']]())
