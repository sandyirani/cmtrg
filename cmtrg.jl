

using TensorOperations
using LinearMaps

include("tensorOpt.jl")
include("updateEnvironment.jl")
include("tests.jl")
include("energy.jl")

X = 20
D = 3
pd = 2
XD2 = X*D*D
X2D2 = X*X*D*D


C = [eye(X) for i=1:4]
Ta = [zeros(X, D, D, X) for i=1:4]
Tb = [zeros(X, D, D, X) for i=1:4]
for k = 1:4
    for i = 1:X
        for j = 1:D
            Ta[k][i,j,j,i] = 1
            Tb[k][i,j,j,i] = 1
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
    (A,B) = initializeAB()
    numIter = 100
    tau = .2
    updateEnvironment(A,B)
    energy = calcEnergy(A,B)
    @show(energy)
    for iter = 1:numIter
        iter%10 == 0 && (tau = 0.2/iter)
        taugate = reshape(expm(-tau * reshape(Htwosite,4,4)),2,2,2,2)
        println("\n iteration = $iter")
        #energies[swp] = sweepFast(m)/N
        (A,B) = applyGateAndUpdate(taugate, RIGHT, A, B)
        updateEnvironment(A,B)
        energy = calcEnergy(A,B)
        @show(energy)
        (A,B) = applyGateAndUpdate(taugate, LEFT, A, B)
        updateEnvironment(A,B)
        energy = calcEnergy(A,B)
        @show(energy)
        (A,B) = applyGateAndUpdate(taugate, UP, A, B)
        updateEnvironment(A,B)
        energy = calcEnergy(A,B)
        @show(energy)
        (A,B) = applyGateAndUpdate(taugate, DOWN, A, B)
        updateEnvironment(A,B)
        energy = calcEnergy(A,B)
        @show(energy)
        #=
        if (iter%10 ==0)
          energy = calcEnergy(A,B)
          @show(energy)
        end
        =#
    end
end

function initializeAB()
    A = zeros(1,1,1,1,2)
    B = zeros(1,1,1,1,2)
    A[1,1,1,1,1] = 1
    B[1,1,1,1,2] = 1
    taugate = reshape(expm(-.2 * reshape(Htwosite,4,4)),2,2,2,2)
    count = 0
    while (count < 4)
        count = 0
        for j = 1:4
            (A,B) = applyGateSU(A,B,taugate)
            newDim = size(A)[2]
            count += (newDim >= D? 1: 0)
            (A, B) = rotateTensors(A,B,UP)
        end
    end
    return(A,B)
end

function applyGateSU(A2,B2,g)
  @tensor begin
    ABg[a,e,f,s1p,b,c,d,s2p] := A2[a,x,e,f,s1]*B2[b,c,d,x,s2]*g[s1,s2,s1p,s2p]
  end
  a = size(ABg)
  ABg = reshape(ABg,a[1]*a[2]*a[3]*pd,a[5]*a[6]*a[7]*pd)
  (U,d,V) = svd(ABg)
  U = U * diagm(d)
  newDim = min(D,length(d))
  U = U[:,1:newDim]
  V = V[:,1:newDim]
  A2p = reshape(U,a[1],a[2],a[3],pd,newDim)
  B2p = reshape(V',newDim,a[5],a[6],a[7],pd)
  A2p = [A2p[i,j,k,s,l] for i=1:a[1], l=1:newDim, j=1:a[2], k=1:a[3], s=1:pd]
  B2p = [B2p[i,j,k,l,s] for j=1:a[5], k=1:a[6], l=1:a[7], i=1:newDim, s=1:pd]
  return(A2p, B2p)
end
