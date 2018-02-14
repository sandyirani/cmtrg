

using TensorOperations
using LinearMaps

include("gridUtilities.jl")
include("tensorUtilities.jl")
include("dmrg.jl")
include("dmrgFast.jl")

X = 20
D = 2
d = 2

A = rand(D, D, D, D, d)
B = rand(D, D, D, D, d)

C = [rand(X, X) for i=1:4]
Ta = [rand(X, X, D, D) for i=1:4]
Tb = [rand(X, X, D, D) for i=1:4]

Ct = [zeros(X, D, D, X) for i=1:4]
Tat = [zeros(X, D, X, D, D, D) for i=1:4]
Tbt = [zeros(X, D, X, D, D, D) for i=1:4]



#Global variables
sz = Float64[0.5 0; 0 -0.5]
sp = Float64[0 1; 0 0]
sm = sp'
Htwosite = reshape(JK(sz,sz) + 0.5 * JK(sp,sm) + 0.5 * JK(sm,sp),2,2,2,2)
# order for Htwosite is s1, s2, s1p, s2p

function mainLoop()
  m = 3
  numIter = 10
  energies = zeros(numIter)
  for iter = 1:numIter
      iter%100 == 0 && (tau = 0.2*100/iter)
      taugate = reshape(expm(-tau * reshape(Htwosite,4,4)),2,2,2,2)
      println("\n iteration = $iter")
      #energies[swp] = sweepFast(m)/N
  end
  energies
end
