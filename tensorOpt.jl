function JK(a,b)	# Julia kron,  ordered for julia arrays; returns matrix
    (a1,a2) = size(a)
    (b1,b2) = size(b)
    reshape(Float64[a[i,ip] * b[j,jp] for i=1:a1, j=1:b1, ip=1:a2, jp=1:b2],a1*b1,a2*b2)
end

function setEnvRight()

  E[1] = Tb[4]
  E[2] = reshape(C[1]*reshape(Tb[1],X,XD2), X, D, D, X)
  E[3] = reshape(reshape(Ta[1],XD2,X)*C[2], X, D, D, X)
  E[4] = Ta[2]
  makeLowerEs(Tb[2],C[3],Tb[3],Ta[3],C[4],Ta[4],B,A)

end

function makeR(B, right)

  if right
    Temp = reshape(E[3],XD2,X)*reshape(E[4],X,XD2)
    Temp = reshape(Temp, X, D, D, D, D, X)
    @tensor begin
      Temp2[x, c, cp, d, dp, y] := Temp[x, a, ap, b, bp, y] * B[a, b, c, d, s] * B[ap, bp, cp, dp, s]
    end
    E5 = E[5]
    @tensor begin
      Temp3[x, d, dp, z] := Temp2[x, c, cp, d, dp, y] * E5[y, c, cp, z]
    end
    R = makeD4Matrix(E[2], Temp3, E[6], E[1])
  else
    Temp = reshape(E[6],X*D^4,X)*reshape(E[1],X,XD2)
    Temp = reshape(Temp, XD2, D, D, D, D, X)
    @tensor begin
      Temp2[x, a, ap, b, bp, y] := Temp[x, c, cp, d, dp, y] * B[a, b, c, d, s] * B[ap, bp, cp, dp, s]
    end
    E2 = E[2]
    @tensor begin
      Temp3[x, b, bp, z] := Temp2[x, a, ap, b, bp, y] * E2[y, a, ap, z]
    end
    R = makeD4Matrix(Temp3, E[3], E[4], E[5])
  end
  return(JK(R,eye(2))

end

function makePsiAB(B, ABp, right)
  Temp = reshape(reshape(E[5],XD2, XD2)*reshape(E[6],XD2,XD2), X, D, D, D, D, X)
  @tensor begin
    Temp2[x, d, e, ap, bp, cp, fp, y, s1p, s2p] := Temp[x, d, dp, e, ep, y]*ABp[ap, bp, cp, dp, ep, fp, s1p, s2p]
  end
  E1 = E[1]
  @tensor begin
    Temp3[x, d, e, ap, bp, cp, f, z, s1p, s2p] := Temp2[x, d, e, ap, bp, cp, fp, y, s1p, s2p] * E1[y, f, fp, z]
  end
  E2 = E[2]
  @tensor begin
    Temp4[x, d, e, a, bp, cp, f, z, s1p, s2p] := Temp3[x, d, e, ap, bp, cp, f, y, s1p, s2p] * E2[y, a, ap, z]
  end
  E3 = E[3]
  @tensor begin
    Temp5[x, d, e, a, b, cp, f, z, s1p, s2p] := Temp4[x, d, e, a, bp, cp, f, y, s1p, s2p] * E3[y, b, bp, z]
  end
  E4 = E[4]
  @tensor begin
    Temp6[a, b, c, d, e, f, s1p, s2p] := Temp5[x, d, e, a, b, cp, f, y, s1p, s2p] * E3[y, c, cp, x]
  end
  Return(Temp6)
end

function getS(B, PsiAB, right)
  if right
      @tensor begin
          S[a, g, e, f, sp1] := PsiAB[a, b, c, d, e, f, s1p, s2p] * B[b, c, d, g, sp2]
      end
  else
      @tensor begin
          S[a, g, e, f, sp1] := PsiAB[a, b, c, d, e, f, s1p, s2p] * B[a, g, e, f, sp1]
      end
  end
  Return(reshape(A,D^4*d))
end

function makeD4Matrix(J, K, L, M)

  j = size(J)
  k = size(K)
  l = size(L)
  Temp = reshape(reshape(J, j[1]*D^2, j[4])*reshape(K, k[1], k[4]*D^2),j[1]*D^4,k[4])
  Temp = Temp*reshape(L,l[1],l[4]*D^2)
  Temp = reshape(Temp,j[1],D,D,D,D,D,D,l[4])
  #This operation is D^8 * X^2
  @tensor begin
    Temp2[a,b,c,d,ap,bp,cp,dp] := Temp[x,a,ap,b,bp,c,cp,y] * M[y,d,dp,x]
  end
  return(reshape(Temp2,D^4, D^4))

end

function makeLowerEs(T1, C1, T2, T3, C2, T4, P1, P2)

  Temp = reshape(T1,XD2,X)*C1*reshape(T2,X,XD2)
  Temp = reshape(Temp, X, D, D, D, D, X)
  @tensor begin
    E5[x, a, ap, d, dp, y] := Temp[x, a, ap, b, bp, y] * P2[a, b, c, d, s] * P2[ap, bp, cp, dp, s]
  end
  E[5] = reshape(E5, X, D, D, XD2)
  Temp = reshape(T3,XD2,X)*C2*reshape(T4,X,XD2)
  Temp = reshape(Temp, X, D, D, D, D, X)
  @tensor begin
    E6[x, b, bp, a, ap, y] := Temp[x, c, cp, d, dp, y] * P1[a, b, c, d, s] * P1[ap, bp, cp, dp, s]
  end
  E[6] = reshape(E6, XD2, D, D, X)

end
