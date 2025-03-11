rm -rf data/*.csv
make clean
make ACCURATE=1 Debug=1
nvprof --metrics warp_execution_efficiency ./slic

