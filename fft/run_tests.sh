for ((p=2; p <= 4; p+=2)) do
    for ((M=6; M <= 12; M++)) do
        echo p: $p , M: $M
        for ((i=0; i<100; i++)) do
            mpirun -np $p python parallel_fft.py $M >> ./tests/p${p}-m${M}.txt
        done
        echo DONE
    done
done
