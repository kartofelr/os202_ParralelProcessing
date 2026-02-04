echo "cores,time" > results.csv
for p in {2..16}
do
    mpiexec --oversubscribe -np $p python bucket.py >> results.csv
done