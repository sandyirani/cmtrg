

using TensorOperations
using LinearMaps

include("tensorOpt.jl")
include("updateEnvironment.jl")

X = 20
D = 3
pd = 2
XD2 = X*D*D
X2D2 = X*X*D*D


C = [rand(X, X) for i=1:4]
Ta = [rand(X, D, D, X) for i=1:4]
Tb = [rand(X, D, D, X) for i=1:4]
for i = 1:4
  for j = 1:X, k = 1:X
    for a = 1:D, b = a+1:D
      Ta[i][j,a,b,k] = .5(Ta[i][j,a,b,k] + Ta[i][j,b,a,k])
      Ta[i][j,b,a,k] = Ta[i][j,a,b,k]
      Tb[i][j,a,b,k] = .5(Tb[i][j,a,b,k] + Tb[i][j,b,a,k])
      Tb[i][j,b,a,k] = Tb[i][j,a,b,k]
    end
  end
end

E = [zeros(X, D, D, X) for i=1:6]
E[5] = zeros(X, D, D, XD2)
E[6] = zeros(XD2, D, D, X)

RIGHT = 1
LEFT = 2
UP = 3
DOWN = 4

#Global variables
sz = Float64[0.5 0; 0 -0.5]
sp = Float64[0 1; 0 0]
sm = sp'
Htwosite = reshape(JK(sz,sz) + 0.5 * JK(sp,sm) + 0.5 * JK(sm,sp),2,2,2,2)
# order for Htwosite is s1, s2, s1p, s2p



function mainLoop()
    A = rand(D, D, D, D, pd) #pd is the particle dimension
    B = rand(D, D, D, D, pd)
    numIter = 1000
    energies = zeros(numIter)
    tau = .2
    for iter = 1:numIter
        @show(size(Bg))
        iter%100 == 0 && (tau = 0.2*100/iter)
        taugate = reshape(expm(-tau * reshape(Htwosite,4,4)),2,2,2,2)
        println("\n iteration = $iter")
        #energies[swp] = sweepFast(m)/N
        @show("apply gate right")
        (A,B) = applyGateAndUpdate(taugate, RIGHT, A, B)
        @show("update environment")
        updateEnvironment(A,B)
        (A,B) = applyGateAndUpdate(taugate, LEFT, A, B)
        updateEnvironment(A,B)
        (A,B) = applyGateAndUpdate(taugate, UP, A, B)
        updateEnvironment(A,B)
        (A,B) = applyGateAndUpdate(taugate, DOWN, A, B)
        updateEnvironment(A,B)
    end
    energies
end
