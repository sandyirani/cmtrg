function calcEnergy(A,B)

  energy = 0

  #Undo
  #setEnv(A, B, RIGHT)
  R = makeR(B,true)
  #R = 0.5*(R + R')
  a = size(A)
  vecA = reshape(A,prod(a))
  norm = vecA'*R*vecA
  (A,B) = renormalizeAll(A,B,norm)

  for dir = 1:1
    (A2, B2) = rotateTensors(A,B,dir)
    (A2p, B2p) = applyGate(A2, B2, Htwosite)
    #Undo
    #setEnv(A2, B2, dir)
    S = makeS(B2,A2p,B2p,true)
    vecA = reshape(A2,prod(size(A2)))
    energy += vecA'*S
  end

  #return(energy/(norm*2))
  return(energy/2)

end

function calcEnergyNoEnv(A,B)

  norm = testNorm(A,B)
  energy = testNorm(A,B,Htwosite)
  return(energy/(norm*2))

end

function renormalizeAll(A,B,norm)
   normFac = abs(norm)
   for j = 1:4
     normFac = sqrt(normFac)
   end
   @show(normFac)
   A = A/sqrt(normFac)
   B = B/sqrt(normFac)
   for j = 1:4
     C[j] = C[j]/normFac
     Tb[j] = Tb[j]/normFac
     Ta[j] = Ta[j]/normFac
   end
   return(A,B)
 end
