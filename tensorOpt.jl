function JK(a,b)	# Julia kron,  ordered for julia arrays; returns matrix
    (a1,a2) = size(a)
    (b1,b2) = size(b)
    reshape(Float64[a[i,ip] * b[j,jp] for i=1:a1, j=1:b1, ip=1:a2, jp=1:b2],a1*b1,a2*b2)
end



function applyGateAndUpdate(g, dir, A, B)
  eps = .00001
  change = 2*eps

  (A2, B2) = rotateTensors(A,B,dir)
  (A2p, B2p) = applyGate(A2, B2, g)
  setEnv(A2, B2, dir)

  @show(calcEnergy(A2,B2))
  @show(calcEnergy(A2p,B2p))

  #For testing
  R = makeR(B2p,true)
  vecA2p = reshape(A2p,prod(size(A2p)))
  normABp = abs(vecA2p'*R*vecA2p)
  @show(normABp)


  oldCostA = 1
  oldCostB = 1
  maxIter = 20
  count = 0

  while( change > eps && count < maxIter)
    count += 1
    R = makeR(B2,true)
    R = stablizeR(R)
    S = makeS(B2,A2p,B2p,true)
    newVecA = getNewAB(R,S)
    A2 = reshape(newVecA,D,D,D,D,pd)
    newCostA = newVecA'*R*newVecA - newVecA'*S - S'*newVecA + normABp
    @show(newCostA/normABp)
    delta = (oldCostA - newCostA)/normABp
    change = abs(delta)

    InverseErrorA = (R*newVecA-S)'*(R*newVecA-S)/(S'*S)
    @show(InverseErrorA)
    if (InverseErrorA > .01)
      @show(InverseErrorA)
    end

    R = makeR(A2,false)
    R = stablizeR(R)
    S = makeS(A2,A2p,B2p,false)
    newVecB = getNewAB(R,S)
    B2 = reshape(newVecB,D,D,D,D,pd)
    newCostB = newVecB'*R*newVecB - newVecB'*S - S'*newVecB + normABp
    @show(newCostB/normABp)
    delta = (oldCostB - newCostB)/normABp
    change = max(change, abs(delta))

    InverseErrorB = sum(abs.(R*newVecB-S))
      @show(InverseErrorB)
    if (InverseErrorB > .01)
      @show(InverseErrorB)
    end

    oldCostA = newCostA
    oldCostB = newCostB
  end
  numberOfIterationsOptAB = count
  #@show(numberOfIterationsOptAB)

  @show(calcEnergy(A2,B2))

  A2 = renormalizeSqrt(A2)
  B2 = renormalizeSqrt(B2)
  #testNorm(A2,B2)
  #testNorm(A2)
  return(rotateTensorsBack(A2,B2,dir))
end



function getNewAB(R, S)
  #return(inv(R)*S)
  return(\(R,S))
end

function applyGate(A2,B2,g)
  @tensor begin
    ABg[a,e,f,s1p,b,c,d,s2p] := A2[a,x,e,f,s1]*B2[b,c,d,x,s2]*g[s1,s2,s1p,s2p]
  end
  ABg = reshape(ABg,D^3*pd,D^3*pd)
  (U,d,V) = svd(ABg)
  U = U * diagm(d)
  newDim = length(d)
  A2p = reshape(U,D,D,D,pd,newDim)
  B2p = reshape(V',newDim,D,D,D,pd)
  A2p = [A2p[i,j,k,s,l] for i=1:D, l=1:newDim, j=1:D, k=1:D, s=1:pd]
  B2p = [B2p[i,j,k,l,s] for j=1:D, k=1:D, l=1:D, i=1:newDim, s=1:pd]
  return(A2p, B2p)
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
  return(Temp3)
end

function mergeLeft(ABup, ABdown)
  Temp = reshape(E[1],XD2,X)*reshape(E[2],X,XD2) #X3 D6
  Temp = reshape(Temp, X, D, D, D, D, X)
  @tensor begin
    Temp2[y, c, cp, b, bp, z] := Temp[y, d, dp, a, ap, z] * ABup[a, b, c, d, s] * ABdown[ap, bp, cp, dp, s] #X2 D8 pd
  end
  E6 = E[6]
  @tensor begin
    Temp3[x, b, bp, z] := E6[x, c, cp, y] * Temp2[y, c, cp, b, bp, z] #X3 D6
  end
  return(Temp3)
end

function makeR(AB, right)

  if right
    Temp = mergeRight(AB, conj.(AB))
    R = makeD4Matrix(Temp, E[6], E[1], E[2])
    s = size(R)
    R = [R[b,c,d,a,bp,cp,dp,ap] for a=1:s[4],b=1:s[1],c=1:s[2],d=1:s[3],ap=1:s[8],bp=1:s[5],cp=1:s[6],dp=1:s[7]]
  else
    Temp = mergeLeft(AB, conj.(AB))
    R = makeD4Matrix(E[5], Temp, E[3], E[4])
    R = [R[c,d,a,b,cp,dp,ap,bp] for a=1:D,b=1:D,c=1:D,d=1:D,ap=1:D,bp=1:D,cp=1:D,dp=1:D]
  end
  r = size(R)
  dim = r[1]*r[2]*r[3]*r[4]
  return(JK(reshape(R,dim,dim),eye(2)))

end

function makeS(AB, AP, BP, right)
  if right
    Temp = mergeRight(AB, BP)
    S = makeD4Matrix(Temp, E[6], E[1], E[2])
    s = size(S)
    S = [S[b,c,d,a,bp,cp,dp,ap] for a=1:s[4],b=1:s[1],c=1:s[2],d=1:s[3],ap=1:s[8],bp=1:s[5],cp=1:s[6],dp=1:s[7]]
    S = reshape(S,prod(s[1:4]),prod(s[5:8]))
    S = reshape(S * reshape(AP,prod(s[5:8]),pd), prod(s[1:4])*pd) # D8 g d
  else
    Temp = mergeLeft(AB, AP)
    S = makeD4Matrix(E[5], Temp, E[3], E[4])
    s = size(S)
    S = [S[c,d,a,b,cp,dp,ap,bp] for a=1:D,b=1:D,c=1:D,d=1:D,ap=1:D,bp=1:D,cp=1:D,dp=1:s[6]]
    S = reshape(S,D^4,D^3*s[6])
    S = reshape(S * reshape(BP,D^3*s[6],pd), D^4*pd) # D8 g d
  end
  return(S)
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
  return(Temp2)

end

function makeLowerEs(T1, C1, T2, T3, C2, T4, P1, P2)

  Temp = reshape(T1,XD2,X)*C1*reshape(T2,X,XD2) # X3 D4
  Temp = reshape(Temp, X, D, D, D, D, X)
  @tensor begin
    E5[x, a, ap, y, d, dp] := Temp[x, b, bp, c, cp, y] * P2[a, b, c, d, s] * P2[ap, bp, cp, dp, s] #X2 D8 d
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

  ap = size(Ap)
  bp = size(Bp)
  if (dir == RIGHT)
    A2 = copy(Ap)
    B2 = copy(Bp)
  elseif (dir == LEFT)
    A2 = [Ap[a,b,c,d,s] for c = 1:ap[3], d = 1:ap[4], a = 1:ap[1], b = 1:ap[2], s = 1:pd]
    B2 = [Bp[a,b,c,d,s] for c = 1:bp[3], d = 1:bp[4], a = 1:bp[1], b = 1:bp[2], s = 1:pd]
  elseif (dir == UP)
    A2 = [Bp[a,b,c,d,s] for b = 1:bp[2], c = 1:bp[3], d = 1:bp[4], a = 1:bp[1], s = 1:pd]
    B2 = [Ap[a,b,c,d,s] for b = 1:ap[2], c = 1:ap[3], d = 1:ap[4], a = 1:ap[1], s = 1:pd]
  elseif (dir == DOWN)
    A2 = [Bp[a,b,c,d,s] for d = 1:bp[4], a = 1:bp[1], b = 1:bp[2], c = 1:bp[3], s = 1:pd]
    B2 = [Ap[a,b,c,d,s] for d = 1:ap[4], a = 1:ap[1], b = 1:ap[2], c = 1:ap[3], s = 1:pd]
  end
  return(A2,B2)

end

function rotateTensorsBack(Ap,Bp,dir)
  if (dir == RIGHT || dir == LEFT)
    return(rotateTensors(Ap,Bp,dir))
  elseif (dir == UP)
    return(rotateTensors(Ap,Bp,DOWN))
  elseif (dir == DOWN)
    return(rotateTensors(Ap,Bp,UP))
  end
end

function stablizeR(R)
    R = 0.5*(R+R')
    return(R)
    r = size(R)
    n = r[1]
    d1 = Int8(round(n/2))
    evn1 = eigs(R;nev=d1,which=:LR,ritzvec=true)
    evn2 = eigs(R;nev=n-d1,which=:SR,ritzvec=true)
    if (length(evn1[1])+length(evn2[1]) < n)
        @show("Not enough eigenvalues")
        return(R)
    end
    evs = zeros(n)
    evs[1:d1] = evn1[1][1:d1]
    evs[d1+1:n] = evn2[1][1:n-d1]
    evsNonNeg = [max(0,evs[j]) for j=1:n]
    eVecs = zeros(n,n)
    eVecs[:,1:d1] = evn1[2][:,1:d1]
    eVecs[:,d1+1:n] = evn2[2][:,1:n-d1]
    #testStabilize = sum(abs.(eVecs*diagm(evs)*eVecs'-R))
    #@show(testStabilize)
    return(eVecs*diagm(evsNonNeg)*eVecs')
end
