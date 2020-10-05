def smoSimple(dataIn, classLabels, C, tolerance, maxIter):
    dataMatrix = mat(dataIn)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))

    bias = 0

    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):

            # Evaluate the model i
            fXi = evaluate(i, alphas, labelMat, dataMatrix, bias)
            Ei = fXi - float(labelMat[i])

            # Check if we can optimize (alphas always between 0 and C)
            if ((labelMat[i] * Ei < -tolerance) and (alphas[i] < C)) or \
                ((labelMat[i] * Ei > tolerance) and (alphas[i] > 0)):

                # Select a random J
                j = selectJrand(i, m)

                # Evaluate the mode j
                fXj = evaluate(j, alphas, labelMat, dataMatrix, bias)
                Ej = fXj - float(labelMat[j])

                # Copy alphas 
                alpha_old_i = alphas[i].copy()
                alpha_old_j = alphas[j].copy()

                # Check how much we can change the alphas
                # L = Lower bound
                # H = Higher bound
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                # If the two correspond, then there is nothing
                # we can really do
                if L == H:
                    print "L is H"
                    continue

                # Calculate ETA
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                        dataMatrix[i, :] * dataMatrix[i, :].T - \
                        dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print "eta is bigger than 0"
                    continue

                # Update J and I alphas
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                # If alpha is not moving enough, continue..
                if abs(alphas[j] - alpha_old_j) < 0.00001:
                    print "Alpha not moving too much.."
                    continue
                # Change alpha I for the exact value, in the opposite
                # direction
                alphas[i] += labelMat[j] * labelMat[i] * \
                        (alpha_old_j - alphas[j])

                # Update bias
                b1 = bias - Ei - labelMat[i] * (alphas[i] - alpha_old_i) * \
                        dataMatrix[i, :] * dataMatrix[i, :].T - \
                        labelMat[j] * (alphas[j]-alpha_old_j) * \
                        dataMatrix[i, :] * dataMatrix[j, :].T

                b2 = bias - Ej - labelMat[i] * (alphas[i] - alpha_old_i) * \
                        dataMatrix[i, :] * dataMatrix[i, :].T - \
                        labelMat[j] * (alphas[j]-alpha_old_j) * \
                        dataMatrix[j, :] * dataMatrix[j, :].T

                # Choose bias to set
                if 0 < alphas[i] and C > alphas[i]:
                    bias = b1
                elif 0 < alphas[j] and C > alphas[j]:
                    bias = b2
                else:
                    bias = (b1 + b2) / 2.0

                # Increment counter and log
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (
                    iter, i, alphaPairsChanged
                )

            if alphaPairsChanged == 0:
                iter += 1
            else:
                iter = 0
            print "Iteration number: %s" % iter

        print alphas[alphas>0]
        print bias
