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

function getRealEnvUpdate()
    T = makeTransferMatrix(A,B)
    testVec = getVec(C[4], Ta[4], Tb[4], C[1])
    for j = 1:5
        newVec = (testVec'*T)'
        changeM = getNormDist(testVec, newVec)
        @show(changeM)
        testVec = newVec
    end
end

function makeTransferMatrix(A,B)
    @show("Making transfer matrix")
    Tb3 = reshape(Tb[3],X,D^2,X)
    Ta3 = reshape(Ta[3],X,D^2,X)
    Tb1 = reshape(Tb[1],X,D^2,X)
    Ta1 = reshape(Ta[1],X,D^2,X)
    Adub = doubleTensor(A)
    Bdub = doubleTensor(B)
    @tensor begin
        Td[x,c1,c2,z]:= Tb3[x,c1,y]*Ta3[y,c2,z]
        T2[z,d2,x,b1,a1,a2] := Td[x,c1,c2,z]*Bdub[a2,b2,c2,d2]*Adub[a1,b1,c1,b2]
        T3[z,d2,d4,x,b1,b3,a3,a4] := T2[z,d2,x,b1,a1,a2]*Bdub[a3,b3,a1,d3]*Adub[a4,d3,a2,d4]
        Tu[w,a4,a3,u] := Tb1[w,a4,y]*Ta1[y,a3,u]
        T4[z,d2,d4,w,x,b1,b3,u] := T3[z,d2,d4,x,b1,b3,a3,a4]*Tu[w,a4,a3,u]
    end
    T = reshape(T4,X^2*D^4,X^2*D^4)
    return(T)
    @show("Done with transfer matrix")
end
