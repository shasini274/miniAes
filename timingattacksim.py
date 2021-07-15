import itertools
import random
from math import sqrt
from scipy.stats import pearsonr
import numpy as np
from pyfinite import ffield
import time


sBox = {
    (0, 0, 0, 0): [1, 1, 1, 0],
    (0, 0, 0, 1): [0, 1, 0, 0],
    (0, 0, 1, 0): [1, 1, 0, 1],
    (0, 0, 1, 1): [0, 0, 0, 1],
    (0, 1, 0, 0): [0, 0, 1, 0],
    (0, 1, 0, 1): [1, 1, 1, 1],
    (0, 1, 1, 0): [1, 0, 1, 1],
    (0, 1, 1, 1): [1, 0, 0, 0],
    (1, 0, 0, 0): [0, 0, 1, 1],
    (1, 0, 0, 1): [1, 0, 1, 0],
    (1, 0, 1, 0): [0, 1, 1, 0],
    (1, 0, 1, 1): [1, 1, 0, 0],
    (1, 1, 0, 0): [0, 1, 0, 1],
    (1, 1, 0, 1): [1, 0, 0, 1],
    (1, 1, 1, 0): [0, 0, 0, 0],
    (1, 1, 1, 1): [0, 1, 1, 1],
       }

sinvBox = {
    (0, 0, 0, 0): [1, 1, 1, 0],
    (0, 0, 0, 1): [0, 0, 1, 1],
    (0, 0, 1, 0): [0, 1, 0, 0],
    (0, 0, 1, 1): [1, 0, 0, 0],
    (0, 1, 0, 0): [0, 0, 0, 1],
    (0, 1, 0, 1): [1, 1, 0, 0],
    (0, 1, 1, 0): [1, 0, 1, 0],
    (0, 1, 1, 1): [1, 1, 1, 1],
    (1, 0, 0, 0): [0, 1, 1, 1],
    (1, 0, 0, 1): [1, 1, 0, 1],
    (1, 0, 1, 0): [1, 0, 0, 1],
    (1, 0, 1, 1): [0, 1, 1, 0],
    (1, 1, 0, 0): [1, 0, 1, 1],
    (1, 1, 0, 1): [0, 0, 1, 0],
    (1, 1, 1, 0): [0, 0, 0, 0],
    (1, 1, 1, 1): [0, 1, 0, 1]
}
#ptxt = np.array([[1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])

k = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])

rcon1 = [0, 0, 0, 1]
rcon2 = [0, 0, 1, 0]


def getNibbleSub(nibble):

    nibbleSub = sBox.get(nibble)
    return nibbleSub

def getGenerateKeySchedule(key, rcon1, rcon2):

    k0, k1, k2 = (np.zeros(shape=(4, 4), dtype=int) for i in range(3))

    for round in range(3):

        if round == 0:
            print(key)
            k0 = key

        elif round == 1:
            try:
                kSub1 = getNibbleSub(tuple(k0[3]))
                for ind in range(4):
                    if ind == 0:
                        try:
                            k1[ind] = k0[ind] ^ kSub1 ^ rcon1
                            print("%s out of four created: %s" % ((ind + 1), k1[ind]))
                        except Exception as ex:
                            print("Error occurred while generating the round key: ", ex)

                    else:
                        try:
                            k1[ind] = k0[ind] ^ k1[ind - 1]
                            print("%s out of four created successfully: %s" % ((ind + 1), k1[ind]))
                        except Exception as ex:
                            print("Error occurred while generating the round key: ", ex)

            except Exception as ex:
                print("Error occurred while generating the %sst round key: " % round, ex)

        elif round == 2:
            try:
                kSub2 = getNibbleSub(tuple(k1[3]))
                for ind in range(4):
                    if ind == 0:
                        try:
                            k2[ind] = k1[ind] ^ kSub2 ^ rcon2
                            print("%s out of four created: %s" % ((ind + 1), k2[ind]))
                        except Exception as ex:
                            print("Error occurred while generating the round key: ", ex)

                    else:
                        try:
                            k2[ind] = k1[ind] ^ k2[ind - 1]
                            print("%s out of four created successfully: %s" % ((ind + 1), k2[ind]))
                        except Exception as ex:
                            print("Error occurred while generating the round key: ", ex)
            except Exception as ex:
                print("Error occurred while generating the %sst round key: " % round, ex)

        else:
            print("Error: Maximum Number of rounds have configured to 3")

    return k0, k1, k2

def getPlaintextKeyAddition(ptxt, rndk0):
    A = ptxt ^ rndk0
    return A

def getShiftRow(etxt):
    etxt[[1, 3], :] = etxt[[3, 1], :]
    return etxt

def getMixColumn(etxt):

    constMat = np.array([[0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 1]])


    lenetxt = int(sqrt(len(etxt)))
    lenconstmat = int(sqrt(len(constMat)))
    mixCol = np.zeros(shape=(4, 4), dtype=int)

    for etxtColNum in range(lenetxt):
        n = etxtColNum * lenetxt
        j = etxtColNum * lenetxt

        for constMatInd in range(lenconstmat):
            m = constMatInd

            waitMat = np.zeros(shape=(2, 4), dtype=int)

            for num in range(lenetxt):

                waitMat[num] = getGfMultiplication(constMat[m], etxt[n])

                n = n + 1
                m = m + 2

            k = 0
            for num2 in range(int(sqrt(len(waitMat)))):

                mixCol[j] = waitMat[k] ^ waitMat[k + 1]

            j = j + 1
            n = n - 2

    return mixCol

def  getGfMultiplication(mat1, mat2):


    ###Unlike in the worked out example here I have used in built GF function##########
    intMat1 = int("".join(str(i) for i in mat1), 2)
    intMat2 = int("".join(str(i) for i in mat2), 2)

    gField = ffield.FField(4)
    mult = gField.Multiply(intMat1, intMat2)

    strmult = '{0:04b}'.format(mult)
    binMat = [int(x) for x in str(strmult)]

    return binMat


def getRoundOneEncTxt(encTxtAd, rndk1):

    for num in range(len(encTxtAd)):
        #print(num)
        encTxtSub[num] = getNibbleSub(tuple(encTxtAd[num]))

    encTxtShift = getShiftRow(encTxtSub)
    encTxtMix = getMixColumn(encTxtShift)
    encTxtRndOne = getPlaintextKeyAddition(encTxtMix, rndk1)

    return encTxtRndOne

def getRoundTwoEncTxt(encTxtR1, rndk2):

    for num in range(len(encTxtR1)):
        #print(num)
        encTxtSub[num] = getNibbleSub(tuple(encTxtR1[num]))

    encTxtShift = getShiftRow(encTxtSub)
    ciphrTxt = getPlaintextKeyAddition(encTxtShift, rndk2)

    return ciphrTxt


if __name__ == '__main__':

    encTxtSub, encTxtShift = (np.zeros(shape=(4, 4), dtype=int) for i in range(2))
    keySchedule = getGenerateKeySchedule(k, rcon1, rcon2)

    rndk0 = keySchedule[0]
    rndk1 = keySchedule[1]
    rndk2 = keySchedule[2]

    n = 4
    #creating the truth table
    lst = [list(i) for i in itertools.product([0, 1], repeat=n)]
    list = np.array(lst)
    lst1 = np.array(lst)
    #print(list)

    b, c, d = (np.zeros(shape=(16, 4), dtype=int) for i in range(3))
    k1, k2, k3, p1, p2, p3 = (np.zeros(shape=(16, 4), dtype=int) for i in range(6))

    N = 10
    avgExecTime = np.zeros(shape=(16, N))

    #Processing to find the actual execution time with the actual secret key
    for k in range(N):
        #obtaining random numbers in the latter part
        for i in range(len(list)):
            b[i] = random.choice(list)
            c[i] = random.choice(list)
            d[i] = random.choice(list)

        plainTxt = np.zeros(shape=(4, 4), dtype=int)

        exec_time = []

        for j in range(len(list)):

            #print(j)
            start_time = time.time()
            plainTxt = np.array([list[j], b[j], c[j], d[j]])
            #print(plainTxt)

            encTxtAd = getPlaintextKeyAddition(plainTxt, rndk0)
            encTxtRnd1 = getRoundOneEncTxt(encTxtAd, rndk1)
            ciphrTxt = getRoundTwoEncTxt(encTxtRnd1, rndk2)
            #print(ciphrTxt)
            time1 = time.time() - start_time
            #print(time1)
            #print(time1)
            avgExecTime[j][k] = avgExecTime[j][k] + time1
            #print(avgExecTime[k][j])
            #exec_time.append(time1)
            #print("--- %s seconds ---" % (time.time() - start_time))

    print(avgExecTime)
    avg = avgExecTime[0:16, N-1]/N
    #print(avg)

    avgExecTime1 = np.zeros(shape=(17, 16))
    avgExecTime1[0, :] = avg


    #################Random Key Generating########
    ####First 4 nibble will be iterate over its truthtable, while the latter is random######
    for n in range(len(list)):
        k1[n] = random.choice(lst1)
        k2[n] = random.choice(lst1)
        k3[n] = random.choice(lst1)
        p1[n] = random.choice(lst1)
        p2[n] = random.choice(lst1)
        p3[n] = random.choice(lst1)

    #######creating different round keys at each instance############
    plainTxtvar = np.zeros(shape=(4, 4), dtype=int)
    exec_time = []

    #######Iterating over 16 different keys##################
    for m in range(16):
        k = np.array([lst1[m], k1[m], k2[m], k3[m]])
        keySchedule = getGenerateKeySchedule(k, rcon1, rcon2)

        rndk0 = keySchedule[0]
        rndk1 = keySchedule[1]
        rndk2 = keySchedule[2]

        ########iterating plain text over eack key instance#####
        for y in range(16):

            plainTxtvar = np.array([lst1[y], p1[y], p2[y], p3[y]])
            #print(plainTxtvar)
            start_time = time.time()

            # print(plainTxt)
            encTxtAd = getPlaintextKeyAddition(plainTxtvar, rndk0)
            encTxtRnd1 = getRoundOneEncTxt(encTxtAd, rndk1)
            ciphrTxt = getRoundTwoEncTxt(encTxtRnd1, rndk2)
            # print(ciphrTxt)
            time1 = time.time() - start_time
            # print(time1)
            # print(time1)
            avgExecTime1[y+1][m] = avgExecTime1[y+1][m] + time1
            #print(y)

    #######check if there is any correlation##########
    for z in range(16):
        corr = pearsonr(avgExecTime1[0], avgExecTime1[z+1])

        print(corr)