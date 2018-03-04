

using TensorOperations
using LinearMaps

include("tensorOpt.jl")
include("updateEnvironment.jl")

X = 20
D = 3
pd = 2
XD2 = X*D*D
X2D2 = X*X*D*D


C = [(rand(X, X)-0.5*ones(X,X))/50 for i=1:4]
Ta = [(zeros(X, D, D, X)-0.5*ones(X, D, D, X))/50 for i=1:4]
Tb = [(zeros(X, D, D, X)-0.5*ones(X, D, D, X))/50 for i=1:4]
for j = 1:4
    C[j][1,1] = 1
    Ta[j][1,1,1,1] = 1
    Tb[j][1,1,1,1] = 1
end
E = [zeros(X, D, D, X)/(.5*X*D) for i=1:6]
E[5] = zeros(X, D, D, XD2)/(.5*XD2)
E[6] = zeros(XD2, D, D, X)/(.5*XD2)

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
    A = (rand(D, D, D, D, pd) - 0.5*ones(D, D, D, D, pd))/50 #pd is the particle dimension
    B = (rand(D, D, D, D, pd) - 0.5*ones(D, D, D, D, pd))/50
    A[1,1,1,1,1] = 1
    B[1,1,1,1,2] = 1
    numIter = 1000
    energies = zeros(numIter)
    tau = .2
    updateEnvironment(A,B)
    for iter = 1:numIter
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
