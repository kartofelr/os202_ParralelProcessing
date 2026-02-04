echo "cores,time,used_cores" > results.csv
for p in {1..16}
do
    echo "working with $p processors"
    mpiexec -np $p python bucket.py >> results.csv
done