/lsc/opt/cuda-12.5/extras/demo_suite/deviceQuery
export PATH=/lsc/opt/cuda-12.5/bin:$PATH
export LD_LIBRARY_PATH=/lsc/opt/cuda-12.5/lib64:$LD_LIBRARY_PATH
nvcc --version
du -sh ~/.* | sort -h
rm -rf ~/.cache/*
du -sh ~/.cache
compute-sanitizer --tool memcheck ./slic