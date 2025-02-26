rm -rf data/*.csv
make clean
make ACCURATE=1 DEBUG=1
./slic
