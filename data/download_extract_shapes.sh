#!/bin/bash

# download
gdown --id 16LE2h4n-T1dEfZ2oN1SrOeq6w33FQW7k --output voxelized_ABC_0.01.zip
wget "https://download.is.tue.mpg.de/faust/MPI-FAUST.zip"
wget "https://datasets.d2.mpi-inf.mpg.de/MultiGarmentNetwork/Multi-Garmentdataset.zip"

# unzip
unzip voxelized_ABC_0.01.zip
unzip MPI-FAUST.zip
unzip Multi-Garmentdataset.zip

# delete .zip file
rm voxelized_ABC_0.01.zip
rm MPI-FAUST.zip
rm Multi-Garmentdataset.zip




 