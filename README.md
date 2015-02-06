# winter 2015 research code
this is my code for work. some of it's quite hacky, and there's a few copy paste files because I really, really don't want to generalize them to work for all subdirectories, especially if I'm not going to be working on those parts anymore.

## system info
- **Operating System**: Ubuntu 14.04 'Trusty' (x86-64)
- **Cinnamon Version**: 2.4.5
- **Linux Kernel**: 3.13.0-45-generic
- **Processor**: Intel Core i5-3210M CPU @ 2.50GHz x 2
- **Memory**: 15.6 GiB

## directories
### matrix-ops
benchmarking simple matrix operations in serial vs parallel. tests include: (square) matrix-matrix addition; (square) matrix-matrix multiplication; matrix-vector multiplication; FFT of a matrix.

### heat-1d
solving the 1-D heat equation in serial and parallel (and benchmarking it too).

### heat-2d
solving the 2-D heat equation ...


## style
I kinda follow the PEP8 style guide, but I ignore: 
- E111
- E127
- E221
- E501

(this doesn't apply to the matrix-ops or heat-1d files, as I didn't start following PEP8 until working with the heat-2d solvers).