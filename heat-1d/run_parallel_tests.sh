for ((p=4; p <= 4; p+=2)) do
  for ((N=4; N <= 4; N++)) do
    for ((m=7; m <= 14; m++)) do
      echo p: $p , N: $N , m: $m
      for ((i=0; i<50; i++)) do
        mpirun -np $p python heat_1d_stepping_mpi.py $m $N $i
        mpirun -np $p python heat_1d_sparse_mpi.py $m $N $i
      done
      echo DONE
    done
  done
done
