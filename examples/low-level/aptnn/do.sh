cell="15.6627 0.0 0.0 0.0 15.6627 0.0 0.0 0.0 15.6627"

data=../../data/
data_sub=${data}/low-level/aptnn

set -eE
trap 'echo "Error: do.sh failed. See output above."; exit 1' ERR

if [ ! -d "${data_sub}" ]; then
    echo "Required data directory not found:"
    echo "  ${data_sub}"
    echo "" 
    echo "Please run the download.sh script located in the examples folder"
    exit 1
fi

# Train the model
mimyria-py train --target apt --train ${data_sub}/training.xyz.zstd --cell ${cell}

# Do prediction on the test set
mimyria-py predict --configs ${data_sub}/test.xyz.zstd --out prediction.xyz.zstd --cell ${cell}
mimyria-py compare --confA prediction.xyz.zstd --confB ${data_sub}/test.xyz.zstd  > rmse.dat

# Now calculate a spectra, using a single provided trajectory
mimyria-py predict --configs ${data}/example-trajectory-pos.xyz.zstd --out pos-and-apt.xyz.zstd --cell ${cell}

# Compute the time correlation function from the APTs and velocities.
# NOTE:  The filelist merges all files in one row to a single trajectory; 
#        thereby it's possible to merge two files on the fly, where on contains positions and apts,
#        and the other one contains the corresponding velocities.
# NOTE2: The filelist can contain an arbitrary number of trajectories, it is therefore possible
#        to process N (in the mimyria paper N=80) NVE trajectories in one go
#        yielding a single time correlation function representing the average over all trajectories
mimyria config.ini --filelist filelist

# water is an isotropic system, thus perform an isotropic average
mimyria-py spectrum-isotropic-average --spectrum_kind ir --inp apt2cf.cf --out apt2cf-iso.cf

# Finally produce the spectrum
mimyria-py corr2spec --corr_in apt2cf-iso.cf --spec_out apt2cf-iso.spec1000 --norm ir --timestep 1 --window_length 1000 --temperature 300 --volume 3842.37624686388

# The directory should now contain a apt2cf-iso.spec1000 that contains the IR spectrum of liquid water
# NOTE:  This IR spectrum can be quite "noisy" because rooting this calculation in a single trajectory only
#        does not provide sufficient statistics; as mentioned above, in the mimyria paper we used 80 trajectories
