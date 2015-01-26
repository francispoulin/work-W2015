for ((N=4; N <= 4; N++)) do
    for ((m=6; m <= 7; m++)) do
        echo N: $N , m: $m
        for ((i=0; i<50; i++)) do
            python heat_1d_sparse.py $N $m >> ./serial-sparse-tests/N${N}-m${m}.txt
            python heat_1d_stepping.py $N $m >> ./serial-stepping-tests/N${N}-m${m}.txt
        done
        echo DONE
    done
done