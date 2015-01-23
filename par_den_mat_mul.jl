function dacmm(i0, i1, j0, j1, k0, k1, A, B, C, n, basecase)
  ## A, B, C are matrices
  ## We compute C = A * B
  if n > basecase
    n = n/2
    dacmm(i0, i1, j0, j1, k0, k1, A, B, C, n, basecase)
    dacmm(i0, i1, j0, j1+n, k0, k1+n, A, B, C, n, basecase)
    dacmm(i0+n, i1, j0, j1, k0+n, k1, A, B, C, n, basecase)
    dacmm(i0+n, i1, j0, j1+n, k0+n, k1+n, A, B, C, n, basecase)
    dacmm(i0, i1+n, j0+n, j1, k0, k1, A, B, C, n, basecase)
    dacmm(i0, i1+n, j0+n, j1+n, k0, k1+n, A, B, C, n, basecase)
    dacmm(i0+n, i1+n, j0+n, j1, k0+n, k1, A, B, C, n, basecase)
    dacmm(i0+n, i1+n, j0+n, j1+n, k0+n, k1+n, A, B, C, n, basecase)
  else
    for i= 1:n, j=1:n, k=1:n
      C[i+k0,k1+j] = C[i+k0,k1+j] + A[i+i0,i1+k] * B[k+j0,j1+j]
    end
  end
end

@everywhere function dacmm_parallel(i0, i1, j0, j1, k0, k1, A, B, C, s, X)
  if s > X
    s = s/2
    lrf = [@spawn dacmm_parallel(i0, i1, j0, j1, k0, k1, A, B, C, s,X),
    @spawn dacmm_parallel(i0, i1, j0, j1+s, k0, k1+s, A, B, C, s,X),
    @spawn dacmm_parallel(i0+s, i1, j0, j1, k0+s, k1, A, B, C, s,X),
    @spawn dacmm_parallel(i0+s, i1, j0, j1+s, k0+s, k1+s, A, B, C, s,X)]
    pmap(fetch, lrf)
    lrf = [@spawn dacmm_parallel(i0, i1+s, j0+s, j1, k0, k1, A, B, C, s,X),
    @spawn dacmm_parallel(i0, i1+s, j0+s, j1+s, k0, k1+s, A, B, C, s,X),
    @spawn dacmm_parallel(i0+s, i1+s, j0+s, j1, k0+s, k1, A, B, C, s,X),
    @spawn dacmm_parallel(i0+s, i1+s, j0+s, j1+s, k0+s, k1+s, A, B, C, s,X)]
    pmap(fetch, lrf)
  else
    for i= 0:(s-1), j=0:(s-1), k=0:(s-1)
      C[i+k0,k1+j] += A[i+i0,i1+k] * B[k+j0,j1+j]
    end
  end
end

n = 1024
A = convert(SharedArray, rand(n,n))
B = convert(SharedArray, rand(n,n))
C = convert(SharedArray, zeros(n,n))

# @time dacmm(0, 0, 0, 0, 0, 0, A, B, C, 64, 32)
# @time A*B
@time A*B
@time dacmm_parallel(1,1,1,1,1,1,A,B,C,n,64)