function calcEnergy(A,B)

  energy = 0

  #Undo
  #setEnv(A, B, RIGHT)
  R = makeR(B,true)
  #R = 0.5*(R + R')
  a = size(A)
  vecA = reshape(A,prod(a))
  norm = vecA'*R*vecA

  for dir = 1:2
    (A2, B2) = rotateTensors(A,B,dir)
    (A2p, B2p) = applyGate(A2, B2, Htwosite)
    #Undo
    #setEnv(A2, B2, dir)
    S = makeS(B2,A2p,B2p,true)
    vecA = reshape(A2,prod(size(A2)))
    energy += vecA'*S
  end

  #(A,B) = renormalizeAll(A,B,norm)

  return(energy/(norm*2))

end

function calcM(A,B)

  energy = 0

  setEnv(A, B, dir)

  R = makeR(B,true)
  a = size(A)
  aHalf = Int64(ceil(a/2))
  sigZbig = JK(eye(aHalf),sigZ)
  vecA = reshape(A,prod(a))
  m1 = vecA'*R*sigZbig*vecA

  R = makeR(A,false)
  b = size(B)
  bHalf = Int64(ceil(b/2))
  sigZbig = JK(eye(bHalf),sigZ)
  vecB = reshape(B,prod(b))
  m2 = vecB'*R*sigZbig*vecB

  return((m1+m2)/2)

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
   @show(norm)
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
