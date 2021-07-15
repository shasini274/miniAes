from math import sqrt
import numpy as np
from config import sBox, sInvBox, key, rcon1, rcon2, constnMat




def getNibbleSub(nibble, subBox):
    """

    :param nibble: This should be a tuple : (1, 1, 1, 0)
    :param subBox: subBox type should be sent over : sBox, sInvBox
    :return: a List corresponding to the tuple : (1, 1, 1, 0): [0, 0, 0, 0]
    """

    nibbleSub = subBox.get(nibble)


    return nibbleSub

def getGenerateKeySchedule(key, rcon1, rcon2):
    """
        Generate 3 keys from this process by using the stored key in the config file
            k0 = key
            k1 = [[k0[0] XOR NibbleSub(k0[3)) XOR Constant_matrix1], [k0[1] XOR k1[0]], [k0[2] XOR k1[1]], [k0[3] XOR k1[2]]]
            k2 = [[k1[0] XOR NibbleSub(k1[3)) XOR Constant_matrix2], [k1[1] XOR k2[0]], [k1[2] XOR k2[1]], [k1[3] XOR k2[2]]]

    """

    k0, k1, k2 = (np.zeros(shape=(4, 4), dtype=int) for i in range(3))

    for round in range(3):

        if round == 0:
            try:
                k0 = np.array(key)
                #print(key)
                print("1st Round key: %s" % k0)
            except Exception as ex:
                print("Error occurred while generating the %sst round key: " % round, ex)
        elif round == 1:
            try:
                print("2nd Round key - k1")
                kSub1 = getNibbleSub(tuple(k0[3]), sBox)
                for ind in range(4):
                    if ind == 0:
                        k1[ind] = k0[ind] ^ kSub1 ^ rcon1
                        print("k1[%s]  created: %s" % ((ind + 1), k1[ind]))
                    else:
                        k1[ind] = k0[ind] ^ k1[ind - 1]
                        print("k1[%s]  created: %s" % ((ind + 1), k1[ind]))
            except Exception as ex:
                print("Error occurred while generating the %sst round key: " % round, ex)
        elif round == 2:
            try:
                print("3rd Round key - k2")
                kSub2 = getNibbleSub(tuple(k1[3]), sBox)
                for ind in range(4):
                    if ind == 0:
                        k2[ind] = k1[ind] ^ kSub2 ^ rcon2
                        print("k2[%s]  created: %s" % ((ind + 1), k2[ind]))

                    else:
                        k2[ind] = k1[ind] ^ k2[ind - 1]
                        print("k2[%s]  created: %s" % ((ind + 1), k2[ind]))
            except Exception as ex:
                print("Error occurred while generating the %sst round key: " % round, ex)
        else:
            print("Error: Maximum Number of rounds have configured to 3")

    return k0, k1, k2

def getKeyAddition(ptxt, rndk0):
    """
         Bitwise XOR perfomance is done
            eg: [0, 1, 0, 0] XOR [0, 1, 1, 0] = [0, 0, 1, 0]
    """

    add = ptxt ^ rndk0
    print("Plain text before addition = %s" % ptxt)
    print("Key Addition with text = %s" % add)
    return add

def getShiftRow(etxt):
    """
        Shift Row is performed as follows:
        list1 = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1], [0, 0, 0, 0]]

        Mat format of List1 is as below
        [1 1 0 0   1 0 0 1]
        [0 0 1 1   0 0 0 0]

        Shifting rows to the either side (right/left) means the same thing
        [1 1 0 0   1 0 0 1]
        [0 0 0 0   0 0 1 1]

        shifted list1:
        list1_1 = [[1, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 1]]
        which means swapping 1 and 3 indexes of the list1
        list1_1 = [list1[0], list1[3], list1[2], list1[1]]
    """
    print("Before Shit Row: %s" % etxt)
    etxt[[1, 3], :] = etxt[[3, 1], :]
    print("After Shift Row: %s" % etxt)
    return etxt

def getMixColumn(etxt):
    """
    :param etxt: is a 4x4 array
           mixcol: is a 4x4 array to store output of the process
        eg :   list1 = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1], [0, 0, 0, 0]]
               mixcol = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        Mat format of List1 is as below
        [1 1 0 0   1 0 0 1]
        [0 0 1 1   0 0 0 0]

        multiplication is done in two stages (GF4 multiplication)
            step1. by extracting the first column of list1 and multiplies it with the constMat --> store in mixcol
                [0 0 1 1   0 0 1 0] * [1 1 0 0]  --> [list1[0] list1[3]] * [etxt[0]] --> [list1[0]*[etxt[0] XOR list1[3]*[etxt[1]] --> mixcol[0]
                [0 0 1 0   0 0 1 1]   [0 0 1 1]      [list1[1] list1[2]]   [etxt[1]]     [list1[1]*[etxt[0] XOR list1[2]*[etxt[1]] --> mixcol[1]

            step2. by extracting the second column of list1 and multiplies it with the constMat --> store in mixcol
                [0 0 1 1   0 0 1 0] * [1 0 0 1] --> [list1[0] list1[3]] * [etxt[2]] --> [list1[0]*[etxt[2] XOR list1[3]*[etxt[3]] --> mixcol[2]
                [0 0 1 0   0 0 1 1]   [0 0 0 0]     [list1[1] list1[2]]   [etxt[3]]     [list1[1]*[etxt[2] XOR list1[2]*[etxt[3]] --> mixcol[3]

        etxtColNum: number of overall iterations --> 2 (because multiplication done in two stages)

    :return:
    """
    constMat = np.array(constnMat)
    lenetxt = int(sqrt(len(etxt)))
    lenconstmat = int(sqrt(len(constMat)))
    mixCol = np.zeros(shape=(4, 4), dtype=int)

    #total number of iterations (decides column 1 or column 2 of the input array)
    for etxtColNum in range(lenetxt):       ## when column 1 --> n index is iterating between 0 and 1
        n = etxtColNum * lenetxt            ## when column 2 --> n index is iterating between 2 and 3
        j = etxtColNum * lenetxt

        for constMatInd in range(lenconstmat):
            m = constMatInd
            waitMat = np.zeros(shape=(2, 4), dtype=int)

            for num in range(lenetxt):
                #wait mat holds the matrix's same row multiplication  values
                # eg: [[constMat[0]*etxt[0]], [constMat[2]*etxt[1]]
                waitMat[num] = getGfMultiplication(constMat[m], etxt[n])
                n = n + 1
                m = m + 2
            k = 0
            for num2 in range(int(sqrt(len(waitMat)))):
                #Performs the XOR operation to the multiplied value of matrix rows
                #mixCol[0] = [constMat[0]*etxt[0]] * [constMat[2]*etxt[1]]
                mixCol[j] = waitMat[k] ^ waitMat[k + 1]
            j = j + 1
            n = n - 2

    return mixCol
def convert_bits_to_index_form(bits):
    """
        Converts bits to an index form.
        eg: [1 0 0 0] => [3]
            [1 0 1 0] => [3,1]
    """

    bits.reverse()
    indexes = []
    for i, bit in enumerate(bits):
        if bit != 0:
            indexes.append(i)
    indexes.sort(reverse=True)
    return indexes


def xor(input1, input2):
    """
        Performs XOR operation on two values which are in the index form.
        eg: [3,1] + [2,0] = [3,2,1,0]   which basically means [1 0 1 0] + [0 1 0 1] = [1 1 1 1]
    """

    output = []
    for i in input1:
        if i in input2:
            input2.remove(i)
            continue
        output.append(i)
    output.extend(input2)
    return output


def reduce(output):
    """
        Performs the functionality of the long division and returns the reduced form.


    """
    poly_of_reduction = [4, 1, 0]   # this represents x4 + x + 1
    while output[0] >= poly_of_reduction[0]:
        multiplier = output[0] - poly_of_reduction[0]                       # similar to dividing x6 by x4. ie, 6 - 4 = 2
        operand = [element + multiplier for element in poly_of_reduction]   # similar to multiplying polynomial_of_red by x2. In this case, adds 2 to each value in [4,1,0] => [6,3,2]  which represents x6+x3+x2

        output = xor(output, operand)                                       # performs XOR operation of the two values

    return output


def convert_to_bits(reduced_output, bit_length):
    """
        Converts the index form back to the bit form
        eg: [3] => [1 0 0 0]
            [2,0] => [0 1 0 1]

    :param reduced_output: index form
    :param bit_length: number of bits of the final output
    :return: result in bit form
    """
    output = [0 for i in range(bit_length)]
    for i in range(len(output)):
        for element in reduced_output:
            if i == element:
                output[i] = 1
    output.reverse()
    return output


def getGfMultiplication(constMat, txtBit):
    """
    :param constMat: 4 bit nibble
    :param txtBit: 4 bit nibble
        constMat, txtBit represents 3rd order polynomials because these are elements of GF4
        eg: [1, 0, 0, 1] => x3 + 0x2 + 0x + 1 => x3 + 1
            [1, 1, 0, 1] => x3 + x2 + 0x + 1 => x3 + x2 + 1

        (x3 + 1)(x3 + x2 + 1) => (x3 + 0x2 + 0x + 1)(x3 + x2 + 0x + 1) => x6 + x5 + 0x4 +  x3
                                                                       =>      0x5+ 0x4 + 0x3 + 0x2
                                                                       =>           0x4 + 0x3 + 0x2 + 0x
                                                                       =>                  x3 +  x2 +  x + 1
                                                       XOR towards down ----------------------------------
                                                                          x6 + x5 + 0x4 + 0x3 + x2 + x + 1
                                                                         [1    1    0     0     1    1   1]
        waitList: Since this is a 3rd order polynomial maximum possible output order would be 6
                  Therefore, List length has set to 6

        GF4 => [0, 0, 0, 0]*[0, 0, 0, 0] = [0, 0, 0, 0]

        Division => (x6 + x5 + X2 + x) / (X4 + x + 1) by synthetic division we get X2 + 1 remainder (x3 + x2)
                     x6 + x5 + X2 + x = [1, 1, 0, 0, 1, 1, 1] >> index form where the values are >> [0, 1, 4, 5, 6] => reverse indexing >> [6, 5, 2, 1, 0]
                     X4 + x + 1 = [1, 0, 0, 1, 1] >> index form where the values are >> [0, 3, 4] => reverse indexing >> [4, 1, 0]

                     similar to dividing x6 by x4. ie, 6 - 4 = 2
                     similar to multiplying polynomial_of_red by x2. In this case, adds 2 to each value in [4,1,0] => [6,3,2]  which represents x6+x3+x2

                     By  [6, 5, 2, 1] XOR [6, 3, 2] => [5, 3, 1, 0]
                     Since x5 >> x4
                     5 - 4 = 1
                     adds 1 to each value in [4,1,0] =>  [5, 2, 0]
                     By  [5, 3, 1, 0] XOR [5, 2, 0] => [3, 2, 1]   => reverse indexing => [0, 1, 2] => converting to binary value of the  index => [1, 1, 1, 0]

         (x3 + 1)(x3 + x2 + 1) = x3 + x2 + 1


    """

    if np.all(constMat == [0, 0, 0, 0]) | np.all(txtBit == [0, 0, 0, 0]):

        output_1 = [0, 0, 0, 0]

    else:
        waitList = []
        for i in range(4):
            waitList.append([0, 0, 0, 0, 0, 0, 0])
            for j, bit in enumerate(constMat):
                if (bit != 0) and (txtBit[i] != 0):
                    waitList[i][j + i] = bit * txtBit[i]

        output = []
        for i in range(7):
            sum = 0
            for j in range(4):
                sum += waitList[j][i]
            output.append(sum % 2)

        # extracting Mod value
        output_index_form = convert_bits_to_index_form(output)  # converts to index form
        reduced_form = reduce(output_index_form) # performs reduction
        output_1 = convert_to_bits(reduced_form, len(constMat))  # convert back to bit form

    return output_1

def getRoundOneEncTxt(encTxtAd, rndk1, sBox):

    for num in range(len(encTxtAd)):
        #print(num)
        encTxtSub[num] = getNibbleSub(tuple(encTxtAd[num]), sBox)
    print("Before Nibble sub: %s" % encTxtAd)
    print("After Nibble Sub: %s" % encTxtSub)

    encTxtShift = getShiftRow(encTxtSub)
    print("Before shiftRow: %s" % encTxtAd)
    print("After shiftRow: %s" % encTxtShift)


    encTxtMix = getMixColumn(encTxtShift)
    print("Mix Column Output: %s" % encTxtMix)
    encTxtRndOne = getKeyAddition(encTxtMix, rndk1)
    print("Rndk1 addition: %s" % encTxtRndOne)


    return encTxtRndOne

def getRoundTwoEncTxt(encTxtR1, rndk2, sBox):

    for num in range(len(encTxtR1)):
        #print(num)
        encTxtSub[num] = getNibbleSub(tuple(encTxtR1[num]), sBox)

    encTxtShift = getShiftRow(encTxtSub)
    ciphrTxt = getKeyAddition(encTxtShift, rndk2)
    print("Cipher Text")
    print(ciphrTxt)

    return ciphrTxt
def getRoundOneDycTxt(dencTxtAd, rndk1, sInvBox):

    dencTxtAd = getKeyAddition(dencTxtAd, rndk1)
    print(dencTxtAd)
    dencTxtMix = getMixColumn(dencTxtAd)
    print(dencTxtMix)
    dencTxtShift = getShiftRow(dencTxtMix)


    for num in range(len(encTxtAd)):

        dencTxtRndOne[num] = getNibbleSub(tuple(dencTxtShift[num]), sInvBox)


    return dencTxtRndOne

if __name__ == '__main__':

    plainTxt = np.array([[1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
    print("Plain Text:")
    encTxtSub, encTxtShift, dencTxtRndOne, dencTxtSub = (np.zeros(shape=(4, 4), dtype=int) for i in range(4))


    keySchedule = getGenerateKeySchedule(key, rcon1, rcon2)

    rndk0 = keySchedule[0]
    rndk1 = keySchedule[1]
    rndk2 = keySchedule[2]

    encTxtAd = getKeyAddition(plainTxt, rndk0)
    encTxtRnd1 = getRoundOneEncTxt(encTxtAd, rndk1, sBox)
    ciphrTxt = getRoundTwoEncTxt(encTxtRnd1, rndk2, sBox)
    print(ciphrTxt)

    dencTxtAd = getKeyAddition(ciphrTxt, rndk2)
    dencTxtShift = getShiftRow(dencTxtAd)

    for num in range(len(encTxtAd)):
        #print(num)
        dencTxtSub[num] = getNibbleSub(tuple(dencTxtShift[num]), sInvBox)

    #print(d111)
    dencTxtRndOne = getRoundOneDycTxt(dencTxtSub, rndk1, sInvBox)
    dencPlainTxt = getKeyAddition(dencTxtRndOne, rndk0)

    print("Decrypt Output: %s" % dencPlainTxt)
