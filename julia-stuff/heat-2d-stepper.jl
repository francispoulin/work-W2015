rand(500,500) * rand(500,500)

Mx = 256
My = 256

# x conditions
x0 = 0.0
xf = 1.0
dx = (xf - x0)/(Mx + 1)
x  = linspace(x0, xf, Mx+2)

# y conditions
y0 = 0.0
yf = 1.0
dy = (yf - y0)/(My + 1)
y  = linspace(y0, yf, My+2)

# temporal conditions
N  = 1000         # time steps
t0 = 0            # start
tf = 300          # end
dt = (tf - t0)/N  # time step size
t  = linspace(t0, tf, N)

# coefficients
k = 0.0002
Kx = 0.02                # PDE coeff for x terms
Ky = 0.01
C  = 1 - 2*(Kx + Ky)

function loopOverArr(arr::Array{Float64, 2})
    rows, cols = size(arr)
    for row = 1:rows
        for col = 1:cols
            arr[row, col] = sin(arr[row, col])
        end
    end
    return arr
end

function applyToArr(arr::Array{Float64, 2})
    return sin(arr)
end