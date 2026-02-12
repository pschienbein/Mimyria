import os
import shlex
import subprocess
from pathlib import Path

from mimyria.autotrain import state_yaml_create_empty_cycle, submit_job, state_get_cmd_line
from mimyria.calculator import get_calculator

from ase.io import iread, write

import numpy as np


def testset(state):
    print('testset called')

    latest = state['cycles'][-1]
    cycle_no = len(state['cycles'])
    target = state['target']

    calculator = get_calculator(state)

    # the state defines how many frames are to be calculated next
    nconf = state['test_set_size']

    # check if the directory already exists
    path = f'cycle{cycle_no}/testset'
    if not os.path.exists(path):
        os.makedirs(path)

        # copy the required frames to the target directory
        configs = list(iread('extracted-configurations.xyz'))[-nconf:]
        write(path + '/configurations.xyz', configs, format='extxyz')

    else:
        configs = list(iread(path + '/configurations.xyz'))

    # check if ALL configurations already contain the target property
    bGotTarget = True
    for conf in configs:
        if target not in conf.arrays:
            bGotTarget = False
            break

    if bGotTarget:
        print(f'Found {target} already in all configurations, skipping calculation...')
        frames2write = configs
    else:
        frames2write = calculator.calculate(state, path, configs)

    if len(frames2write) != nconf:
        print('Number of configurations calculated does not match the number of configurations requested! Likely something went wrong while evaluating the electronic structure.')
        print('NOTE: Also make sure that the calculation was not cancelled by a time out.')
        return False

    write(path + '/test.xyz', frames2write, format='extxyz')

    # if reached this, update the state
    latest['testset']['status'] = 'done'

    # continue the process!
    return True


def refdata(state):
    print('refdata called')

    latest = state['cycles'][-1]
    cycle_no = len(state['cycles'])
    target = state['target']

    calculator = get_calculator(state)

    print(cycle_no)

    # the state defines how many frames are to be calculated next
    nconf = state['sn']['batchsize']
    beg = (cycle_no - 1) * nconf
    end = cycle_no * nconf

    # exceeding limits?
    if end > state['sn']['maxconf']:
        print(f'reached maximum configurations of {beg} as requested, exiting...')
        return False

    # check if the directory already exists
    path = f'cycle{cycle_no}/refdata'
    if not os.path.exists(path):
        os.makedirs(path)

        # copy the required frames to the target directory
        configs = []
        for i, atoms in enumerate(iread('extracted-configurations.xyz')):
            if i >= beg:
                configs.append(atoms)
            if i == end - 1:
                break
        write(path + '/configurations.xyz', configs, format='extxyz')

    else:
        configs = list(iread(path + '/configurations.xyz'))

    # check if ALL configurations already contain the target property
    bGotTarget = True
    for conf in configs:
        if target not in conf.arrays:
            bGotTarget = False
            break

    if bGotTarget:
        print(f'Found {target} already in all configurations, skipping calculation...')
        frames2write = configs
    else:
        frames2write = calculator.calculate(state, path, configs)

    if len(frames2write) != nconf:
        print('Number of configurations calculated does not match the number of configurations requested! Likely something went wrong while evaluating the electronic structure')
        return False

    write(path + '/training.xyz', frames2write, format='extxyz')

    # if reached this, update the state
    latest['refdata']['status'] = 'done'

    # continue the process!
    return True


def training(state):
    print('training called')

    latest = state['cycles'][-1]
    cycle_no = len(state['cycles'])

    # check if the directory already exists
    path = f'cycle{cycle_no}/training'
    print(path)
    if not os.path.exists(path):
        os.makedirs(path)

        # accumulate training data from all cycles
        with open(path + '/training.xyz', 'w') as fout:
            for f in Path('.').glob('cycle*/refdata/training.xyz'):
                with open(f) as fin:
                    fout.write(fin.read())

        # create the run script:
        print(state['target'])
        with open(path + '/run.sh', 'w') as fout:
            # additional cmd arguments:
            cmd_predict = state_get_cmd_line(state, 'predict')
            cmd_train = state_get_cmd_line(state, 'train')

            fout.write(f'''
mimyria-py train --out model.mym --train training.xyz --target {state['target']} {cmd_train}
if [ $? -ne 0 ]; then exit 1; fi
mimyria-py predict --model model.mym --configs ../../cycle1/testset/test.xyz --out testset-prediction.xyz {cmd_predict}
if [ $? -ne 0 ]; then exit 1; fi
touch FIN
            ''')

        # do the job
        submit_job(state, path, 'gpu', bWait=True)

    # check if the process is finished;
    # this is just to avoid running the script twice at the same time
    if not os.path.exists(path + '/FIN'):
        print('Training process running, exiting....')
        return False

    # check if the model has been created
    if not os.path.exists(path + '/model.mym'):
        print('model.mym was not created, likely something went wrong!')
        return False

    # now do the evaluation of the predicted APTs
    cmd = f'mimyria-py compare --target {state["target"]} --confA testset-prediction.xyz --confB ../../cycle1/testset/test.xyz --scatterout scatter-{{symbol}}.dat --out rmse.dat'
    result = subprocess.run(shlex.split(cmd), cwd=path, capture_output=True, text=True)

    # write the obtained RMSEs to file
    with open(path + '/rmse.dat') as fin:
        for line in fin:
            lastline = line

    with open('rmse.dat', 'a') as fout:
        fout.write(f'{cycle_no} {lastline}')

    state['cycles'].append(state_yaml_create_empty_cycle())
    latest['training']['status'] = 'done'

    # check exit condition
    return cycle_no < state['max_cycles']


def predict_full(state, cycle_no, bCont : bool = False):
    path = f'cycle{cycle_no}'
    if not os.path.exists(path):
        raise FileNotFoundError("Cycle not existing")

    bFoundExistingFile = False

    path = f'cycle{cycle_no}/predict_full'
    realpath = os.path.realpath(path)

    width = int(np.ceil(np.log10(len(state['trajectories']))))

    # check if the directory is existing
    if not os.path.exists(path):
        os.makedirs(path)
    elif not bCont:
        raise FileExistsError("Directory {path} is already existing")

    # additional cmd arguments:
    cmd_predict = state_get_cmd_line(state, 'predict')

    # create the run command
    # with open(path + '/run.sh', 'w') as fout,\
    with open(path + '/command.list', 'w') as fout,\
         open(path + '/run.sh', 'w') as fout_run,\
         open(path + '/filelist', 'w') as fout_flist:

        fout_run.write('''
id=${SLURM_ARRAY_TASK_ID}
cmd=$(sed -n "${id}p" command.list)

if [[ -z "${cmd// }" ]]; then
  echo "No command found on line ${id}"
  exit 1
fi

echo "Running ${cmd}"
bash -lc "${cmd}"
        ''')

        cell_array = np.array(state['cell'])
        variable_cell = (cell_array.ndim == 3)

        for i, traj in enumerate(state['trajectories']):
            infile = os.path.realpath(traj)

            if variable_cell:
                cell_arg = ' '.join(map(str, cell_array[i].flatten()))
            else:
                cell_arg = ' '.join(map(str, cell_array.flatten()))

            target_fn = f'predict-{i:0{width}d}.xyz.zstd'
            bInclude = True

            if bCont:
                # check if the file is already existing:
                if os.path.exists(f'{path}/{target_fn}'):
                    bFoundExistingFile = True
                    bInclude = False

            if bInclude:
                fout.write(f'mimyria-py predict --model ../training/model.mym --configs {infile} --cell {cell_arg} --out {target_fn} {cmd_predict}\n')
                # fout.write('if [ $? -ne 0 ]; then exit; fi\n')

            # write this entry to the filelist
            fout_flist.write(f'{realpath}/{target_fn}')
            if 'velocities' in state:
                vel_fn = state['velocities'][i]
                fout_flist.write(f', {vel_fn}')

                if 'velocities_properties' in state:
                    props = state['velocities_properties'][i]
                    fout_flist.write(f' ({props})')
            fout_flist.write('\n')

        fout.write('touch FIN')

    if bCont and bFoundExistingFile:
        # print a warning
        print('WARNING: Running in continuing mode. Note that existing files are NOT continued, they are NOT checked for completeness!')

    # this executes the commands given above in path as wd
    N = len(state['trajectories'])
    submit_job(state, path, 'gpu', bWait=True, array=[1, N])
