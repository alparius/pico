#!/bin/bash

mkdir -p static
cd static

# Download utility files and other constants

# smplx faces
wget https://keeper.mpdl.mpg.de/f/e1b97e01671e4b92852f/?dl=1 --max-redirect=2 --trust-server-names
# smpl to smplx conversion
wget https://keeper.mpdl.mpg.de/f/21d408011a5d45f1a602/?dl=1 --max-redirect=2 --trust-server-names

cd ..
