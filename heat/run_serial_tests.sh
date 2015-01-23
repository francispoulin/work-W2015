for ((N=5; N <= 7; N++)) do
    for ((m=2; m <= 3; m++)) do
        echo N: $N , m: $m
        for ((i=0; i<100; i++)) do
            python heat_1d_sparse.py $N $m >> ./serial-sparse-tests/N${N}-m${m}.txt
            python heat_1d_stepping.py $N $m >> ./serial-stepping-tests/N${N}-m${m}.txt
        done
        echo DONE
    done
done