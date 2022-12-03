a = [
[[[1,2,3],[1,2,3],[1,2,3]], 1],
[[[2,2,3],[1,2,3],[1,2,3]], 2],
[[[3,2,3],[1,2,3],[1,2,3]], 3]
]

list_ = []
for c in range(len(a[0])):
    for s in range(len(a)):
        for val in a[s][c]:
            max = max
    list_.append(max)

b = [j[i] for j in a for i in range(len(a[0]))]

a = list(list(zip(*a))[:-1][0])

print(a)