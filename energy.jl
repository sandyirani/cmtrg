function calcEnergy(A,B)

  energy = 0

  for dir = 1:4
    (A2, B2) = rotateTensors(A,B,dir)
    (A2p, B2p) = applyGate(A2, B2, Htwosite)
    setEnv(A2, B2, dir)


    S = makeS(B2,A2p,B2p,true)
    vecA = reshape(A2,D^4*pd)
    energy += vecA'*S
    @show(energy)
  end

  setEnv(A, B, RIGHT)
  R = makeR(B,true)
  #R = 0.5*(R + R')
  vecA = reshape(A,D^4*pd)
  norm = vecA'*R*vecA
  @show(norm)
  @show(vecA'*vecA)
  @show(trace(R))

  return(energy/(norm*2))

end
