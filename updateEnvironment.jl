function updateEnvironment()

  (C[4], Tb[4], Ta[4], C[1]) = genericUpdate(Ta[3],C[4],Ta[4],Tb[4],C[1],Tb[1],A,B)
  (C[4], Ta[4], Tb[4], C[1]) = genericUpdate(Tb[3],C[4],Tb[4],Ta[4],C[1],Ta[1],B,A)

  (A2,B2) = rotateTensors(A,B,UP)
  (C[1], Ta[1], Tb[1], C[2]) = genericUpdate(Tb[4],C[1],Tb[1],Ta[1],C[2],Ta[2],A2,B2)
  (C[1], Tb[1], Ta[1], C[2]) = genericUpdate(Ta[4],C[1],Ta[1],Tb[1],C[2],Tb[2],B2,A2)

  (A2,B2) = rotateTensors(A,B,LEFT)
  (C[2], Tb[2], Ta[2], C[3]) = genericUpdate(Ta[1],C[2],Ta[2],Tb[2],C[3],Tb[3],A,B)
  (C[2], Ta[2], Tb[2], C[3]) = genericUpdate(Tb[1],C[2],Tb[2],Ta[2],C[3],Ta[3],B,A)

  (A2,B2) = rotateTensors(A,B,DOWN)
  (C[3], Ta[3], Tb[3], C[4]) = genericUpdate(Tb[2],C[3],Tb[3],Ta[3],C[4],Ta[4],A2,B2)
  (C[3], Tb[3], Ta[3], C[4]) = genericUpdate(Ta[2],C[3],Ta[3],Tb[3],C[4],Tb[4],B2,A2)

end

function genericUpdate(TAd, Cld, TAl, TBl, Clu, TBu, Adub, Bdub)

  Cld1 = reshape(reshape(TAd,XD2,X)*Cld,X,XD2)
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
  evn = eigs(MZ;nev=X,ritzvec=true)
  Z = evn[2][:,1:X]

  MQ1 = reshape(reshape(TBl,XD2,X)*Clu*reshape(TBu,X,XD2),X,D^2,D^2,X)
  @tensor begin
    Q1[x,b,c,y] := MQ[x,d,a,y]*Adub[a,b,c,d]
  end
  Q1 = reshape(Q1,XD2,XD2)

  MQ4 = reshape(reshape(TAd,XD2,X)*Cld*reshape(TAl,X,XD2),X,D^2,D^2,X)
  @tensor begin
    Q4[x,b,a,y] := MQ[x,c,d,y]*Bdub[a,b,c,d]
  end
  Q4 = reshape(Q1,XD2,XD2)

  MW = conj.(Q1'*Q1) + Q4+Q4'
  evn = eigs(MW;nev=X,ritzvec=true)
  W = evn[2][:,1:X]

  newClu = Z'*Clu1
  newCld = Cld1*Z
  newTAl = reshape(Z'*reshape(TBl1,XD2,X*D^4),XD2,XD2)
  newTAl = reshape(newTAl1*W,X,D,D,X)
  newTBl = reshape(W'*reshape(TAl1,XD2,X*D^4),XD2,XD2)
  newTBl = reshape(newTBl1*Z,X,D,D,X)

  Return(newCld, newTAl, newTBl, newClu)

end
