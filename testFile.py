from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

import kmedoids

# 3 points in dataset
data = np.array([[1,1],
                [2,2],
                [10,10]])

# distance matrix
D = pairwise_distances(data, metric='euclidean')

file = open('distances.txt','r')

f_dat = file.readlines()

dists = []

for line in f_dat:
    values = line.split(',')
    d_arr = []
    for value in values:
        d_arr.append(float(value))

    dists.append(d_arr)



array = np.array(dists)
num = len(array)

print num
print len(array[0])

maxx = 0
minx = 10
count = 0
sum = 0
for i in range(num):
    for j in range(num):
        if array[i][j] != 0:
            sum += array[i][j]
            count += 1
            if(array[i][j] > maxx):
                maxx = array[i][j]
            if(array[i][j] < minx):
                minx = array[i][j]


print minx, maxx

for i in range(num):
    for j in range(num):
        if array[i][j] != 0:
            array[i][j] = (array[i][j] - minx)/maxx

D = np.matrix(array)
print D
n = len(D)
# split into 2 clusters
def cost(d_mat,M,C):
    k = len(M)
    costs = []
    for i in range(k):
        costs.append(0)
    for c_i in range(k):
        for i in C[c_i]:
            costs[c_i] += d_mat[M[c_i],i]

    return np.sum(costs)

M, C = kmedoids.kMedoids(D, n, 2)

for i in range(1000):
    t_M, t_C = kmedoids.kMedoids(D, n, 2)
    if(cost(D,t_M,t_C) < cost(D,M,C)):
        M = t_M
        C = t_C

print('medoids:')
for point_idx in M:
    # print( data[point_idx] )
    print point_idx


# print('')

print('clustering result:')

for label in C:
    for point_idx in C[label]:
        print 'label ' + str(label) + ': ' + str(point_idx)

print C[0]
print C[1]
print len(C[0])
sad = 0
happy = 0
for point in C[0]:
    if point > 445:
        sad += 1
    else:
        happy += 1
print happy, sad
print len(C[1])
sad = 0
happy = 0
for point in C[1]:
    if point > 445:
        sad += 1
    else:
        happy += 1
print happy, sad

