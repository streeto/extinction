#!/bin/bash

opts="--lll=4000 --fixk --outdir=figures"

cubes="../../cubes.200/K0708_synthesis_eBR_v01_q036.d13c512.ps03.k2.mC.CCM.Bgsd61.fits \
../../cubes.200/K0708_synthesis_eBR_v20_q036.d13c512.ps03.k2.mC.CCM.Bgsd61.fits \
../../cubes_1.4/K0708_synthesis_eBR_v20_q042.d14512.ps03.k2.mC.CCM.Bgsd61.fits \
../../cubes.200/K0925_synthesis_eBR_v01_q036.d13c512.ps03.k2.mC.CCM.Bgsd61.fits \
../../cubes.200/K0925_synthesis_eBR_v20_q036.d13c512.ps03.k2.mC.CCM.Bgsd61.fits \
../../cubes_1.4/K0925_synthesis_eBR_v20_q042.d14512.ps03.k2.mC.CCM.Bgsd61.fits"

for K in $cubes; do
	python ../extinction.py $opts $K
done
