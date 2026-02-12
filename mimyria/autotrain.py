import shutil
import shlex
import subprocess
import time
import re


def state_yaml_create_empty_cycle():
    return {
            'refdata': {'status': 'pending'},
            'training': {'status': 'pending'},
           }


# state: the state dict
# script: either being predict or train
def state_get_cmd_line(state, script):
    cmd = ''
    key = f'cmd_{script}'
    if key in state:
        for k in state[key]:
            cmd += f'--{k} {state[key][k]} '

    return cmd

# Currently only supports slurm queue environments
def is_job_running(jobid):
    result = subprocess.run(['squeue', '-j', jobid],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    return result.returncode == 0


def is_job_running_in_path(path):
    try:
        with open(path + '/job.id') as fin:
            return is_job_running(fin.read())
    except OSError:
        # if the file is not existing, treat as "not running"
        return False


def get_job_exit_code(jobid):
    result = subprocess.run(['sacct', '-j', str(jobid), '--format=JobID,ExitCode,State', '--parsable2', '--noheader'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=True)

    array_exit_codes = {}
    single_exit_code = None

    for line in result.stdout.splitlines():
        if not line:
            continue

        jid, exit_code = line.split("|", 1)

        # Extract exit status
        try:
            code = int(exit_code.split(":")[0])
        except ValueError:
            continue

        # Case 1: non-array job
        if jid == str(jobid):
            single_exit_code = code

        # Case 2: array task (jobid_<index>)
        else:
            m = re.fullmatch(rf"{jobid}_(\d+)", jid)
            if m:
                array_exit_codes[int(m.group(1))] = code

    # No records at all → invalid job ID
    if single_exit_code is None and not array_exit_codes:
        return None

    # Array job
    if array_exit_codes:
        exit_code_list = [
            array_exit_codes[i] for i in sorted(array_exit_codes)
        ]
        return array_exit_codes

    # Non-array job
    return single_exit_code


def get_job_exit_code_in_path(path):
    try:
        with open(path + '/job.id') as fin:
            return get_job_exit_code(fin.read())
    except OSError:
        return None


# Submits a job to the queue for execution
# Currently only supports slurm queue environments
# NOTE: bForceLocalExecution bypasses the queue and runs the job ON THE CURRENT NODE!
# array: Should be a pair with the first and last index
def submit_job(state, path, kind, bWait=True, bForceLocalExecution=False, array=None):
    # check for submit decorator
    if state[f'use_submit_decorator_{kind}'] and not bForceLocalExecution:
        # copy the gpu runscript
        shutil.copy(f'run-{kind}.sub', path)

        cmd = 'sbatch '
        if array is not None:
            cmd += f'--array={array[0]}-{array[1]} '
        cmd += f'run-{kind}.sub'
        # result = subprocess.run(['sbatch', f'run-{kind}.sub'], cwd=path, capture_output=True, text=True)
        result = subprocess.run(shlex.split(cmd), cwd=path, capture_output=True, text=True)
        print(result.stdout)

        if result.returncode != 0:
            print('ERROR submitting job!')
            print(result.stderr)
            return False

        # if successful, this is returned by sbatch
        # Submitted batch job XXXXX
        jobid = result.stdout.split()[-1]
        with open(path + '/job.id', 'w') as fout:
            fout.write(jobid)

        if bWait:
            # poll until the slurm job completes
            # NOTE: This loop exits if the call fails
            # it's not up to this routine to check if the job ran successfully!
            while is_job_running(jobid):
                time.sleep(60)

        # ok on this end, continue
        return True

    else:
        # NOTE: This is always executed in bWait == True mode
        with open(path + '/run.out', 'w') as out, open(path + '/run.err', 'w') as err:
            proc = subprocess.run(['bash', 'run.sh'], cwd=path, stdout=out, stderr=err)
            return proc.returncode == 0
