for i in `seq 1 1000`;
do
  echo $i
  mpirun -np 4 python parallel_matvec.py >> tests/p4-m13.txt
done
echo DONE
