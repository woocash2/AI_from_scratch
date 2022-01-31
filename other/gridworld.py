import numpy as np
import operator


N = int(input())     # grid rows
M = int(input())     # grid cols
Ascore = 10.
Bscore = 5.
gamma = 0.9

grid = np.zeros((N, M), dtype=int)
current_pts = []


def place_point():
    while True:
        P = (np.random.randint(0, N), np.random.randint(0, M))
        if P not in current_pts:
            current_pts.append(P)
            grid[P[0]][P[1]] = int(len(current_pts))
            return P


A = place_point()
B = place_point()
Apr = place_point()
Bpr = place_point()

print(grid)


def neighbours(x, y, policy):   # not for A, B
    return {
        'L': (x, y - 1, policy[x][y]['L']),
        'R': (x, y + 1, policy[x][y]['R']),
        'U': (x - 1, y, policy[x][y]['U']),
        'D': (x + 1, y, policy[x][y]['D'])
    }.copy()


def in_range(x, y):
    return 0 <= x < N and 0 <= y < M


def evaluate(policy, values, iters):
    for i in range(1, iters + 1):
        for x in range(N):
            for y in range(M):
                values[i % 2][x][y] = 0.
                if (x, y) == A:
                    values[i % 2][x][y] = Ascore + gamma * values[(i + 1) % 2][Apr[0]][Apr[1]]
                elif (x, y) == B:
                    values[i % 2][x][y] = Bscore + gamma * values[(i + 1) % 2][Bpr[0]][Bpr[1]]
                else:
                    neibs = neighbours(x, y, policy)
                    for direction, p in neibs.items():  # p = (xnei, ynei, prob entering nei)
                        if not in_range(p[0], p[1]):
                            values[i % 2][x][y] += p[2] * (-1. + gamma * values[(i + 1) % 2][x][y])         # bump into wall and continue from x, y
                        else:
                            values[i % 2][x][y] += p[2] * (0. + gamma * values[(i + 1) % 2][p[0]][p[1]])    # 0


def policy_improvement(policy, values):
    new_policy = []
    for x in range(N):
        new_policy.append([])
        for y in range(M):
            new_policy[x].append({'L': 0., 'R': 0., 'U': 0., 'D': 0.})
            if (x, y) in [A, B]:
                continue
            neibs = neighbours(x, y, policy)
            best = -1e18
            act = 'L'
            for direction, n in neibs.items():
                if not in_range(n[0], n[1]):    # vals[x][y] - 1 < vals[x][y]
                    if best < -1 + gamma * values[0][x][y]:
                        best = -1 + gamma * values[0][x][y]
                        act = direction
                    continue
                if best < values[0][n[0]][n[1]]:
                    act = direction
                    best = values[0][n[0]][n[1]]    # at least for one neib, values[neib] >= values[x, y]
            new_policy[x][y][act] = 1.
    return new_policy


def policy_iteration(iters, eval_iters):
    values = np.zeros((2, N, M))
    policy = []
    for x in range(N):
        policy.append([])
        for y in range(M):
            policy[x].append({'L': 0.25, 'R': 0.25, 'U': 0.25, 'D': 0.25})

    for i in range(iters):
        evaluate(policy, values, eval_iters)
        policy = policy_improvement(policy, values)
        print('After', i + 1)
        print('Grid')
        print(grid)
        print('Evaluation')
        for x in range(N):
            for y in range(M):
                print(round(values[0][x][y], 2), end=' ')
            print()
        print('Policy')
        for x in range(N):
            for y in range(M):
                c = max(policy[x][y].items(), key=operator.itemgetter(1))[0]
                print(c, end='')
            print()
        print()


policy_iteration(10, 1000)
