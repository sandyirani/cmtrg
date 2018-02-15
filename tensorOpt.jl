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
      Temp2[x, c, cp, b, bp, y] := Temp[x, d, dp, a, ap, y] * B[a, b, c, d, s] * B[ap, bp, cp, dp, s]
    end
    Temp = reshape(reshape(Temp2, XD2, XD2) * reshape(E[5],XD2,X), X, D, D, D, D, X)
    R = makeD4Matrix(E[2], Temp, E[6], E[1])
  else
    Temp = reshape(E[6],X*D^3,X)*reshape(E[1],X,XD2)
    Temp = reshape(Temp, XD2, D, D, D, D, X)
    @tensor begin
      Temp2[x, a, ap, b, bp, y] := Temp[x, c, cp, d, dp, y] * B[a, b, c, d, s] * B[ap, bp, cp, dp, s]
    end
    Temp = reshape(reshape(Temp2, X*D^4, XD2) * reshape(E[2],XD2,X), XD2, D, D, D, D, X)
    R = makeD4Matrix(E[3], E[4], E[5], Temp2)
  end 

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
