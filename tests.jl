function testRotateTensors()
  A = rand(D,D,D,D,pd)
  B = rand(D,D,D,D,pd)

  (Aup, Bup) = rotateTensors(A,B,UP)
  (Aleft1,Bleft1) = rotateTensors(Aup,Bup,UP)
  (Aleft2,Bleft2) = rotateTensors(A,B,LEFT)
  (A2,B2) = rotateTensors(Aleft2,Bleft2,LEFT)
  (Adown1, Bdown1) = rotateTensors(Aup,Bup,LEFT)
  (Adown2, Bdown2) = rotateTensors(A,B,DOWN)

  @show(sum(abs.(Aup-A)))
  @show(sum(abs.(Aup-B)))
  @show(sum(abs.(Adown1-Adown2)))
  @show(sum(abs.(Bdown1-Bdown2)))
  @show(sum(abs.(Aleft1-Aleft2)))
  @show(sum(abs.(Bleft1-Bleft2)))
  @show(sum(abs.(A2-A)))
  @show(sum(abs.(B2-B)))

end

function initABdub()
  (A,B) = initializeAB()
  tau = .2

  taugate = reshape(expm(-tau * reshape(Htwosite,4,4)),2,2,2,2)
  #updateEnvironment(A,B)

  Adub = [zeros(D^2,D^2,D^2,D^2) for i=1:4]
  Bdub = [zeros(D^2,D^2,D^2,D^2) for i=1:4]
  for dir=1:4
    (A2,B2) = rotateTensors(A,B,dir)
    (Adub[dir], Bdub[dir]) = (doubleTensor(A2),doubleTensor(B2))
  end
  return(Adub,Bdub)
end

function testEnv()
  (Adub,Bdub) = initABdub()
  @show("AB initialized")
  change = 1
  while(change > .005)
    oldVec = getVec(C[4], Ta[4], Tb[4], C[1])
    (C[4], Tb[4], Ta[4], C[1], w1) = genericUpdate(Ta[3],C[4],Ta[4],Tb[4],C[1],Tb[1],Adub[RIGHT],Bdub[RIGHT])
    (C[4], Ta[4], Tb[4], C[1], w2) = genericUpdate(Tb[3],C[4],Tb[4],Ta[4],C[1],Ta[1],Bdub[RIGHT],Adub[RIGHT])
    newVec = getVec(C[4], Ta[4], Tb[4], C[1])
    change = getNormDist(oldVec, newVec)
    @show(getNormDist(oldVec, newVec))
  end
end
