function updateEnvironment(A,B)

  EPS = .001
  change = 1

  Adub = [zeros(D^2,D^2,D^2,D^2) for i=1:4]
  Bdub = [zeros(D^2,D^2,D^2,D^2) for i=1:4]
  for dir=1:4
    (A2,B2) = rotateTensors(A,B,dir)
    (Adub[dir], Bdub[dir]) = (doubleTensor(A2),doubleTensor(B2))
  end

  while(change > EPS)
    oldVec = getVec(C[4], Ta[4], Tb[4], C[1])
    (C[4], Tb[4], Ta[4], C[1]) = genericUpdate(Ta[3],C[4],Ta[4],Tb[4],C[1],Tb[1],Adub[RIGHT],Bdub[RIGHT])
    (C[4], Ta[4], Tb[4], C[1]) = genericUpdate(Tb[3],C[4],Tb[4],Ta[4],C[1],Ta[1],Bdub[RIGHT],Adub[RIGHT])
    newVec = getVec(C[4], Ta[4], Tb[4], C[1])
    change = getNormDist(oldVec, newVec)
    @show("Left")
    @show(getNormDist(oldVec, newVec))

    oldVec = getVec(C[2],Ta[2],Tb[2],C[3])
    (C[2], Tb[2], Ta[2], C[3]) = genericUpdate(Ta[1],C[2],Ta[2],Tb[2],C[3],Tb[3],Adub[LEFT],Bdub[LEFT])
    (C[2], Ta[2], Tb[2], C[3]) = genericUpdate(Tb[1],C[2],Tb[2],Ta[2],C[3],Ta[3],Bdub[LEFT],Adub[LEFT])
    newVec = getVec(C[2],Ta[2],Tb[2],C[3])
    change = max(change, getNormDist(oldVec, newVec))
    @show("Right")
    @show(getNormDist(oldVec, newVec))
  end

  change = 1
  while(change > EPS)
    oldVec = getVec(C[1],Tb[1],Ta[1],C[2])
    (C[1], Ta[1], Tb[1], C[2]) = genericUpdate(Tb[4],C[1],Tb[1],Ta[1],C[2],Ta[2],Adub[UP],Bdub[UP])
    (C[1], Tb[1], Ta[1], C[2]) = genericUpdate(Ta[4],C[1],Ta[1],Tb[1],C[2],Tb[2],Bdub[UP],Adub[UP])
    newVec = getVec(C[1],Tb[1],Ta[1],C[2])
    change = getNormDist(oldVec, newVec)
    @show("Top")
    @show(getNormDist(oldVec, newVec))

    oldVec = getVec(C[3],Tb[3],Ta[3],C[4])
    (C[3], Ta[3], Tb[3], C[4]) = genericUpdate(Tb[2],C[3],Tb[3],Ta[3],C[4],Ta[4],Adub[DOWN],Bdub[DOWN])
    (C[3], Tb[3], Ta[3], C[4]) = genericUpdate(Ta[2],C[3],Ta[3],Tb[3],C[4],Tb[4],Bdub[DOWN],Adub[DOWN])
    newVec = getVec(C[3],Tb[3],Ta[3],C[4])
    change = max(change, getNormDist(oldVec, newVec))
    @show("Bottom")
    @show(getNormDist(oldVec, newVec))
  end

end

function getVec(C1,Ta,Tb,C2)
  M = C1*reshape(Ta,X,XD2)
  M = reshape(M,XD2,X)*reshape(Tb,X,XD2)
  M = reshape(M,X*D^4,X)*C2
  return(reshape(M,X^2*D^4))
end

function getVecBig(C1,Ta,Tb,C2)
  M = C1*reshape(Ta,XD2,X*D^4)
  M = reshape(M,XD2,XD2)*reshape(Tb,XD2,X*D^4)
  M = reshape(M,X*D^4,XD2)*C2
  return(reshape(M,X^2*D^4))
end

function getNormDist(v1,v2)
  v1 = v1/sqrt(v1'*v1)
  v2 = v2/sqrt(v2'*v2)
  return(sqrt(max(0,2-v1'*v2-v2'*v1)))
end

function genericUpdate(TAd, Cld, TAl, TBl, Clu, TBu, Adub, Bdub)

  Cld1 = reshape(reshape(TAd,XD2,X)*Cld,X,D^2,X)
  Cld1 = [Cld1[a,b,c] for a=1:X,c=1:X,b=1:D^2]
  Cld1 = reshape(Cld1,X,XD2)

  TAl = reshape(TAl,X,D^2,X)
  @tensor begin
    TBl1[x,c,b,y,a] := TAl[x,d,y]*Bdub[a,b,c,d]
  end
  TBl1 = reshape(TBl1,XD2,D^2,XD2)

  TBl = reshape(TBl,X,D^2,X)
  @tensor begin
    TAl1[x,c,b,y,a] := TBl[x,d,y]*Adub[a,b,c,d]
  end
  TAl1 = reshape(TAl1,XD2,D^2,XD2)

  Clu1 = reshape(Clu*reshape(TBu,X,XD2),XD2,X)

  MZ = Clu1*Clu1' + conj.(Cld1'*Cld1)
  evn = eigs(MZ;nev=X,which=:LM,ritzvec=true)
  Z = evn[2][:,1:X]

  MQu = reshape(reshape(TBl,XD2,X)*Clu*reshape(TBu,X,XD2),X,D^2,D^2,X)
  @tensor begin
    Qu[x,c,y,b] := MQu[x,d,a,y]*Adub[a,b,c,d]
  end
  Qu = reshape(Qu,XD2,XD2)

  MQd = reshape(reshape(TAd,XD2,X)*Cld*reshape(TAl,X,XD2),X,D^2,D^2,X)
  @tensor begin
    Qd[x,b,y,a] := MQd[x,c,d,y]*Bdub[a,b,c,d]
  end
  Qd = reshape(Qd,XD2,XD2)

  MW = conj.(Qd'*Qd) + Qu*Qu'
  evn = eigs(MW;nev=X,ritzvec=true)
  W = evn[2][:,1:X]

  newClu = renormalize(Z'*Clu1)
  newCld = renormalize(Cld1*Z)
  newTBl = reshape(Z'*reshape(TBl1,XD2,X*D^4),XD2,XD2)
  newTBl = renormalize(reshape(newTBl*W,X,D,D,X))
  newTAl = reshape(W'*reshape(TAl1,XD2,X*D^4),XD2,XD2)
  newTAl = renormalize(reshape(newTAl*Z,X,D,D,X))

  w1 = getVecBig(Cld1,TBl1,TAl1,Clu1)
  w2 = getVec(newCld, newTBl, newTAl, newClu)
  @show("Truncation error")
  dist = getNormDist(w1,w2)
  @show(dist)

  return(newCld, newTBl, newTAl, newClu)

end

function doubleTensor(AB)
    conjAB = conj.(AB)
    @tensor begin
        ABdub[a,ap,b,bp,c,cp,d,dp] := AB[a,b,c,d,s]*conjAB[ap,bp,cp,dp,s]
    end
    return(reshape(ABdub,D^2,D^2,D^2,D^2))
end

function renormalize(T)
  t = size(T)
  aveT = sum(abs.(T))/sum(t)
  T = T/(aveT*sqrt(sum(t)))
  return(T)
end
