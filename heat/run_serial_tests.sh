for ((N=2; N <= 4; N++)) do
    for ((m=7; m <= 16; m++)) do
        echo N: $N , m: $m
        for ((i=0; i<50; i++)) do
            python heat_1d_sparse.py $N $m >> ./tests/ser-spar/N${N}-m${m}.txt
            python heat_1d_stepping.py $N $m >> ./tests/ser-step/N${N}-m${m}.txt
        done
        echo DONE
    done
done