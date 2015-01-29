for ((N=4; N <= 4; N++)) do
    for ((m=7; m <= 14; m++)) do
        echo N: $N , m: $m
        for ((i=0; i<50; i++)) do
            python heat_1d_sparse.py $m $N $i
            python heat_1d_stepping.py $m $N $i
        done
        echo DONE
    done
done