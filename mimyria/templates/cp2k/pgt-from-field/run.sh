#!/bin/bash

# check that critical variables are set
if [ -z "${MPIRUN}" ]; then
    echo "Error: The variable MPIRUN is not set by run-cpu.sub. It should contain the full MPIRUN string including the numbers of processes and potential pinning, such as \"mpirun -np 8\"" >&2
    exit 1
fi

if [ -z "${CP2K}" ]; then
    echo "Error: The variable CP2K is not set by run-cpu.sub. It should contain the executable name of cp2k including a potential path, such as \"cp2k.popt\"" >&2
    exit 1
fi

# get the number of configurations
N=$(grep Prop configurations.xyz | wc -l)

i=${SLURM_ARRAY_TASK_ID}
dir=$(printf "%05d" "$i")

mkdir "$dir"
rsync -a --exclude='*.out' --exclude='*/' ./ $dir
cd "$dir"

ln -sf ../force_eval.inc 

sed -i "s/FIRST_SNAPSHOT.*/FIRST_SNAPSHOT ${i}/g" *.inp
sed -i "s/LAST_SNAPSHOT.*/LAST_SNAPSHOT ${i}/g" *.inp

for f in run-*.inp
do
  # NOTE: this needs to be adjusted for different systems
  outfn=${f%.inp}.out
  ${MPIRUN} "${CP2K}" "${f}" >> "${outfn}"
  ret=$?

  if [ $ret -ne 0 ]; then
    # the execution of CP2k failed, exit with error
    exit 1
  fi

  grep 'SCF run NOT converged' ${outfn}        
  if [ $? -eq 0 ]; then
    # the SCF did not converge, exit with error
    exit 2
  fi
done

rm *wfn*

exit 0
