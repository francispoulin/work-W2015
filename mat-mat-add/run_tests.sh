for ((p=2; p <= 4; p+=2)) do
    for ((M=13; M <= 13; M++)) do
        echo p: $p , M: $M
        for ((i=0; i<100; i++)) do
            echo $i
            mpirun -np $p python parallel_matmatadd.py $M >> ./tests/p${p}-m${M}.txt
        done
        echo DONE
    done
done