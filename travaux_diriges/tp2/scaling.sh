echo "cores,time" > results.csv
for p in {1..16}
do
    echo "working with $p processors"
    mpiexec -n $p python mandelbrot_vec_parallel.py >> results.csv || exit 1
done