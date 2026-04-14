cell="15.6627 0.0 0.0 0.0 15.6627 0.0 0.0 0.0 15.6627"

data=../../data/
data_sub=${data}/low-level/pgtnn

set -eE
trap 'echo "Error: do.sh failed. See output above."; exit 1' ERR

if [ ! -d "${data_sub}" ]; then
    echo "Requramaned data directory not found:"
    echo "  ${data_sub}"
    echo "" 
    echo "Please run the download.sh script located in the examples folder"
    exit 1
fi

# setup the autotrainer 
# --trajectories specify as many trajectories as you like / have
# --velocities specify the corresponding velocity files, if velocities are not contained in the trajectory file
# --velocities_property_string The exemplary velocity file contains velocities instead of positions and no information in the comment line (default CP2k output); with the velocities_properties_string we override how mimyria should interpret the columns in that file in this case reading it as velocities.
mimyria-py autotrain_setup \
    --trajectories ../../data/example-trajectory-pos.xyz.zstd \
    --velocities ../../data/example-trajectory-vel.xyz.zstd \
    --velocities_property_string "Properties=species:S:1:vel:R:3" \
    --cell ${cell} \
    --target pgt \
    --test_set_size 10 \
    --max_cycles 3


# Now execute the training
# NOTE: autotrain_step will run until the calculation is fully finished; 
#       if an error occurs, it can be called again and will resume at that point where it crashed.
mimyria-py autotrain_step

# Finally predict for all trajectories you specified during setup
# NOTE: Now only for the final cycle
mimyria-py autotrain_predict_full --cycle 3

# Compute the time correlation function from the APTs and velocities.
# NOTE: The filelist is created by the auto trainer and contains all trajectory files!
mimyria config.ini --filelist cycle3/predict_full/filelist

# water is an isotropic system, thus perform an isotropic average
mimyria-py spectrum-isotropic-average --spectrum_kind raman --inp pgt2cf.cf --out pgt2cf-iso.cf

# Finally produce the spectrum
mimyria-py corr2spec --corr_in pgt2cf-iso.cf --spec_out pgt2cf-iso.spec1000 --norm raman --timestep 1 --window_length 1000 --temperature 300 --volume 3842.37624686388

