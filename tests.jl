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
        #@show(changeM)
        testVec = newVec
    end
end

function testNorm(A,B)
    Aconj = conj.(A)
    Bconj = conj.(B)
    @tensor begin
        top[a,b,c,d,e,f,s1,s2] := A[a,x,e,f,s1]*B[b,c,d,x,s2]
        bottom[ap,bp,cp,dp,ep,fp,s1p,s2p] := Aconj[ap,xp,ep,fp,s1p]*Bconj[bp,cp,dp,xp,s2p]
        tb[a,b,c,d,e,f,ap,bp,cp,dp,ep,fp] := top[a,b,c,d,e,f,s1,s2]*bottom[ap,bp,cp,dp,ep,fp,s1,s2]
    end
    s = size(tb)
    dim = s[1]*s[2]*s[3]*s[4]*s[5]*s[6]
    #@show(trace(reshape(tb,dim,dim)))
end

function testNorm(A)
    Aconj = conj.(A)
    @tensor begin
        tb[a,b,c,d,ap,bp,cp,dp] := A[a,b,c,d,s]*Aconj[ap,bp,cp,dp,s]
    end
    s = size(tb)
    dim = s[1]*s[2]*s[3]*s[4]
    #@show(trace(reshape(tb,dim,dim)))
end

function calcNormAlt(A,B)

  vLeft = getVec(C[4],Ta[4],Tb[4],C[1])
  M = makeTransferMatrix(A,B)
  vRight = getVecRight(C[3],Tb[2],Ta[2],C[2])
  #@show(vLeft'*vLeft)
  #@show(vRight'*vRight)
  #n = length(vLeft)
  #vLeftRand = rand(n)
  #vLeftRand = vLeftRand/(vLeftRand'*vLeftRand)*(vLeft'*vLeft)
  #vRightRand = rand(n)
  #vRightRand = vRightRand/(vRightRand'*vRightRand)*(vRight'*vRight)
  #@show(vLeftRand'*M*vRightRand)
  return(vLeft'*M*vRight)

end

function testConverge()
    n = X^2*D^4
    M = rand(n,n)
    v = rand(n)
    vOld = v
    numT = 100
    for j = 1:numT
        vOld = v
        v = (v'*M)'
        v = v/sqrt(v'*v)
        #@show(getNormDist(v,vOld))
    end
end


function testTransfer(M)

    #M = makeTransferMatrix(A,B)
    #M = rand(X^2*D^4,X^2*D^4)
    #v = getVec(C[3],Tb[3],Ta[3],C[4])
    M = M*1000000
    v = rand(size(M)[1])
    vOld = v
    numT = 100
    for j = 1:numT
        vOld = v
        v = (v'*M)'
        @show(sqrt(v'*v))
        v = v/sqrt(v'*v)
        v = (v'*M)'
        v = v/sqrt(v'*v)
        @show(getNormDist(v,vOld))
    end
end

function makeTransferMatrix(A,B)
    #@show("Making transfer matrix")
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
    #@show("Done with transfer matrix")
end

function makeTransferMatrixMid(A,B)
    @show("Making transfer matrix")
    Aconj = conj.(A)
    Bconj = conj.(B)
    @tensor begin
      Atop[d,dp,c,cp,b,bp] := A[a,b,c,d,s]*Aconj[a,bp,cp,dp,s]
      Btop[d,dp,c,cp,b,bp] := B[a,b,c,d,s]*Bconj[a,bp,cp,dp,s]
      Abot[d,dp,a,ap,b,bp] := A[a,b,c,d,s]*Aconj[ap,bp,c,dp,s]
      Bbot[d,dp,a,ap,b,bp] := B[a,b,c,d,s]*Bconj[ap,bp,c,dp,s]
    end
    Atop = reshape(Atop,D^4,D^2)
    Btop = reshape(Btop,D^2,D^4)
    Top = reshape(Atop*Btop,D^2,D^2,D^2,D^2)
    Abot = reshape(Abot,D^4,D^2)
    Bbot = reshape(Bbot,D^2,D^4)
    Bot = reshape(Abot*Bbot,D^2,D^2,D^2,D^2)
    @tensor begin
      T[a1,a,d1,d] := Top[a,b,c,d]*Bot[a1,b,c,d1]
    end
    @show("Done with transfer matrix")
    return(reshape(T,D^4,D^4))
end

function makeTransferMatrixSides()
    @show("Making transfer matrix")
    Tb3 = reshape(Tb[3],X,D^2,X)
    Ta3 = reshape(Ta[3],X,D^2,X)
    Tb1 = reshape(Tb[1],X,D^2,X)
    Ta1 = reshape(Ta[1],X,D^2,X)
    @tensor begin
        Td[x,c1,c2,z]:= Tb3[x,c1,y]*Ta3[y,c2,z]
        Tu[w,c2,c1,u] := Tb1[w,c2,y]*Ta1[y,c1,u]
        T4[z,w,x,u] := Td[x,c1,c2,z]*Tu[w,c2,c1,u]
    end
    T = reshape(T4,X^2,X^2)
    return(T)
    @show("Done with transfer matrix")
end
