function JK(a,b)	# Julia kron,  ordered for julia arrays; returns matrix
    (a1,a2) = size(a)
    (b1,b2) = size(b)
    reshape(Float64[a[i,ip] * b[j,jp] for i=1:a1, j=1:b1, ip=1:a2, jp=1:b2],a1*b1,a2*b2)
end

eps = .001

function applyGateAndUpdate(g, dir)

  (A2, B2) = rotateTensors(A,B,dir)

  (A2p, B2p) = applyGate(A2, B2, g)

  setEnv(A2, B2, dir)

  change = 2*eps

  while( change > eps )
    R = makeR(B2,true)
    S = makeS(B2,A2p,B2p,true)
    newA = reshape(getNewAB(R,S),D,D,D,D,d)
    change = sum(abs.(newA-A2))
    A2 = newA

    R = makeR(A2,false)
    S = makeS(A2,A2p,B2p,false)
    newB = reshape(getNewAB(R,S),D,D,D,D,d)
    change = max(sum(abs.(newB-B2)), change)
    B2 = newB
  end

  (A, B) = rotateTensorsBack(A2,B2,dir)

end

function applyGate(A2,B2,g)
  @tensor begin
    ABg[a,e,f,a1,b,c,d,s2p] := A2[a,x,e,f,s1]*B2[b,c,d,x,s2]*g[s1,s2,s1p,s2p]
  end
  ABg = reshape(ABg,D^4*d,D^4*d)
  (U,d,V) = svd(ABg)
  U = U * diagm(d)
  newDim = length(d)
  A2p = reshape(U,D,D,D,d,newDim)
  B2p = reshape(U,newDim,D,D,D,d)
  A2p = [A2p[i,m,j,k,l] for i=1:D, j=1:D, k=1:D, l=1:d, m=1:newDim]
  B2p = [B2p[j,k,l,i,m] for i=1:newDim, j=1:D, k=1:D, l=1:D, m=1:d]
  Return(A2p, B2p)
end

function setEnv(A2,B2,dir)

  if (dir == RIGHT)
    E[1] = Tb[4]
    E[2] = reshape(C[1]*reshape(Tb[1],X,XD2), X, D, D, X)
    E[3] = reshape(reshape(Ta[1],XD2,X)*C[2], X, D, D, X)
    E[4] = Ta[2]
    makeLowerEs(Tb[2],C[3],Tb[3],Ta[3],C[4],Ta[4],B2,A2)
  elseif (dir == LEFT)
    E[1] = Tb[2]
    E[2] = reshape(C[3]*reshape(Tb[3],X,XD2), X, D, D, X)
    E[3] = reshape(reshape(Ta[3],XD2,X)*C[4], X, D, D, X)
    E[4] = Ta[4]
    makeLowerEs(Tb[4],C[1],Tb[1],Ta[1],C[2],Ta[2],B2,A2)
  elseif (dir == UP)
    E[1] = Ta[1]
    E[2] = reshape(C[2]*reshape(Ta[2],X,XD2), X, D, D, X)
    E[3] = reshape(reshape(Tb[2],XD2,X)*C[3], X, D, D, X)
    E[4] = Tb[3]
    makeLowerEs(Ta[3],C[4],Ta[4],Tb[4],C[1],Tb[1],B2,A2)
  elseif (dir == DOWN)
    E[1] = Ta[3]
    E[2] = reshape(C[4]*reshape(Ta[4],X,XD2), X, D, D, X)
    E[3] = reshape(reshape(Tb[4],XD2,X)*C[1], X, D, D, X)
    E[4] = Tb[1]
    makeLowerEs(Ta[1],C[2],Ta[2],Tb[2],C[3],Tb[3],B2,A2)
  end

end

function mergeRight(ABup, ABdown)
  Temp = reshape(E[3],XD2,X)*reshape(E[4],X,XD2)  #X2 D4
  Temp = reshape(Temp, X, D, D, D, D, X)
  @tensor begin
    Temp2[x, c, cp, d, dp, y] := Temp[x, a, ap, b, bp, y] * ABup[a, b, c, d, s] * ABdown[ap, bp, cp, dp, s] #X2 D8 d
  end
  E5 = E[5]
  @tensor begin
    Temp3[x, d, dp, z] := Temp2[x, c, cp, d, dp, y] * E5[y, c, cp, z] # X3 D6
  end
  Return(Temp3)
end

function mergeLeft(ABup, ABdown)
  Temp = reshape(E[6],X*D^4,X)*reshape(E[1],X,XD2) #X3 D6
  Temp = reshape(Temp, XD2, D, D, D, D, X)
  @tensor begin
    Temp2[x, a, ap, b, bp, y] := Temp[x, c, cp, d, dp, y] * ABup[a, b, c, d, s] * ABdown[ap, bp, cp, dp, s] #X2 D8 d
  end
  E2 = E[2]
  @tensor begin
    Temp3[x, b, bp, z] := Temp2[x, a, ap, b, bp, y] * E2[y, a, ap, z] #X3 D4
  end
  Return(Temp3)
end

function makeR(AB, right)

  if right
    Temp = mergeRight(AB, AB)
    R = makeD4Matrix(E[2], Temp, E[6], E[1])
  else
    Temp = mergeLeft(AB, AB)
    R = makeD4Matrix(Temp3, E[3], E[4], E[5])
  end
  return(JK(R,eye(2))

end

function makeS(AB, AP, BP, right)
  if right
    Temp = mergeRight(AB, BP)
    S = makeD4Matrix(E[2], Temp, E[6], E[1])
    ap = size(AP)
    S = reshape(S * reshape(AP,ap[1]*ap[2]*ap[3]*ap[4],ap[5]), ap[1]*ap[2]*ap[3]*ap[4]*ap[5]) # D8 g d
  else
    Temp = mergeLeft(AB, AP)
    S = makeD4Matrix(Temp3, E[3], E[4], E[5])
    Bp = size(BP)
    S = reshape(S * reshape(AP,bp[1]*bp[2]*bp[3]*bp[4],bp[5]), bp[1]*bp[2]*bp[3]*bp[4]*bp[5]) # D8 g d
  end
  Return(S)
end

function makeD4Matrix(J, K, L, M)

  j = size(J)
  k = size(K)
  l = size(L)
  Temp = reshape(reshape(J, j[1]*j[2]*j[3], j[4])*reshape(K, k[1], k[4]*k[3]*k[2]),j[1]*j[2]*j[3]*k[2]*k[3],k[4])
  Temp = Temp*reshape(L,l[1],l[4]*l[2]*l[3])
  Temp = reshape(Temp,j[1],j[2],j[3],k[2],k[3],l[2],l[3],l[4])
  # X2 D8 g
  @tensor begin
    Temp2[a,b,c,d,ap,bp,cp,dp] := Temp[x,a,ap,b,bp,c,cp,y] * M[y,d,dp,x]
  end
  return(reshape(Temp2,D^4, D^4))

end

function makeLowerEs(T1, C1, T2, T3, C2, T4, P1, P2)

  Temp = reshape(T1,XD2,X)*C1*reshape(T2,X,XD2)
  Temp = reshape(Temp, X, D, D, D, D, X)
  @tensor begin
    E5[x, a, ap, d, dp, y] := Temp[x, a, ap, b, bp, y] * P2[a, b, c, d, s] * P2[ap, bp, cp, dp, s] #X2 D8 d
  end
  E[5] = reshape(E5, X, D, D, XD2)
  Temp = reshape(T3,XD2,X)*C2*reshape(T4,X,XD2) # X3 D4
  Temp = reshape(Temp, X, D, D, D, D, X)
  @tensor begin
    E6[x, b, bp, a, ap, y] := Temp[x, c, cp, d, dp, y] * P1[a, b, c, d, s] * P1[ap, bp, cp, dp, s] #X2 D8 d
  end
  E[6] = reshape(E6, XD2, D, D, X)

end

function rotateTensors(Ap,Bp,dir)

  if (dir == RIGHT)
    A2 = copy(Ap)
    B2 = copy(Bp)
  elseif (dir == LEFT)
    A2 = [Ap[c,d,a,b,s] for a = 1:D for b = 1:D for c = 1:D for d = 1:D for s = 1:d]
    B2 = [Bp[c,d,a,b,s] for a = 1:D for b = 1:D for c = 1:D for d = 1:D for s = 1:d]
  elseif (dir == UP)
    A2 = [Bp[b,c,d,a,s] for a = 1:D for b = 1:D for c = 1:D for d = 1:D for s = 1:d]
    B2 = [Ap[b,c,d,a,s] for a = 1:D for b = 1:D for c = 1:D for d = 1:D for s = 1:d]
  elseif (dir == DOWN)
    A2 = [Bp[d,a,b,c,s] for a = 1:D for b = 1:D for c = 1:D for d = 1:D for s = 1:d]
    B2 = [Ap[d,a,b,c,s] for a = 1:D for b = 1:D for c = 1:D for d = 1:D for s = 1:d]
  end
  Return(A2,B2)

end

function rotateTensorsBack(Ap,Bp,dir)
  if (dir == RIGHT || dir == LEFT)
    Return(rotateTensors(Ap,Bp,dir))
  elseif (dir == UP)
    Return(rotateTensors(Ap,Bp,DOWN))
  elseif (dir == DOWN)
    Return(rotateTensors(Ap,Bp,UP))
  end
end
