
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

sInvBox = {
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

key = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]

rcon1 = [0, 0, 0, 1]
rcon2 = [0, 0, 1, 0]

constnMat = [[0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 1]]