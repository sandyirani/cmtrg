function mainLoop()
  m = 3
  numIter = 10
  energies = zeros(numIter)
  for iter = 1:numIter
      iter%100 == 0 && (tau = 0.2*100/iter)
      taugate = reshape(expm(-tau * reshape(Htwosite,4,4)),2,2,2,2)
      println("\n iteration = $iter")
      #energies[swp] = sweepFast(m)/N
  end
  energies
end
