python3 train.py unet-R-12fl/baseline 0 5 > catch.txt &
PID0=$!
python3 train.py unet-R-12fl/baseline-10 1 5 > catch.txt &
PID1=$!
python3 train.py unet-R-12fl/baseline-90 2 5 > catch.txt &
PID2=$!
python3 train.py unet-R-12fl/map-GT-10 3 5 > catch.txt &
PID3=$!
wait $PID0 $PID1 $PID2 $PID3
python3 train.py unet-R-12fl/map-GT-25 0 5 > catch.txt &
PID0=$!
python3 train.py unet-R-12fl/map-GT-100 1 5 > catch.txt &
PID1=$!
python3 train.py unet-R-12fl/simple-GT-100 2 5 > catch.txt &
PID2=$!
python3 train.py unet-R-12fl/map-LT-10 3 5 > catch.txt &
PID3=$!
wait $PID0 $PID1 $PID2 $PID3
python3 train.py unet-R-12fl/map-LT-25 0 5 > catch.txt &
PID0=$!
python3 train.py unet-R-12fl/map-LT-100 1 5 > catch.txt &
PID1=$!
python3 train.py unet-R-12fl/simple-LT-100 2 5 > catch.txt &
PID2=$!
wait $PID0 $PID1 $PID2
python3 write_results.py unet-R-12fl/results.txt unet-R-12fl/baseline unet-R-12fl/baseline-10 unet-R-12fl/baseline-90 unet-R-12fl/map-GT-10 unet-R-12fl/map-GT-25 unet-R-12fl/map-GT-100 unet-R-12fl/simple-GT-100 unet-R-12fl/map-LT-10 unet-R-12fl/map-LT-25 unet-R-12fl/map-LT-100 unet-R-12fl/simple-LT-100
python3 summarize.py unet-R-12fl/results.txt