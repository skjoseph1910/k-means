from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import random
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pyclustering.cluster.kmedoids as km
from sklearn.metrics import silhouette_score

def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape
    
    if k > n:
        raise Exception('too many medoids')
    
    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs,cs = np.where(D==0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r,c in zip(rs,cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(len(invalid_medoid_inds)))
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])
    
    # create a copy of the array of medoid indices
    Mnew = np.copy(M)
    
    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

# return results
    return M, C



data = np.array([[9,8],
                 [3,3],
                 [10,10], [9,9], [4,4]])

# distance matrix
D = pairwise_distances(data, metric='euclidean')
#print(D)


df = pd.read_excel('hello.xlsx')
array2=np.array(df)
kmeanref = ['student1', 'student2', 'student3', 'student4', 'student5', 'student6', 'student7','student8','student9','student10','student11', 'student12', 'student13','student14','student15','student16','student17', 'student18','student19','student20']
'''
size=len(array)
print(size)

#df = pd.DataFrame(array)

#df=pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index)

'''
print(array2)
array = pairwise_distances(array2, metric = 'euclidean')

#print(array)
#print(D)

clustering = DBSCAN(eps=26, min_samples=3, metric = 'euclidean').fit(array)

#print(clustering.labels_)
#print(clustering)
y= clustering.fit_predict(array)
print((y))
avg = silhouette_score(array, y)
print(avg)

plt.scatter(array[:,0], array[:,1], c=y, cmap='Paired')


#plt.show()

#db = DBSCAN(eps = .5, min_samples = 2, metric = 'euclidean')
#db.fit_predict(distance_matrix)


M, C = kMedoids(array, 6)


print('medoids:')
for point_idx in M:
    print(kmeanref[point_idx] )

list = []
print('')
print('clustering result:')
for label in C:
    for point_idx in C[label]:
        print('label {0}:　{1}'.format(label, kmeanref[point_idx]))

### automate this process
list = [5, 4, 1, 4, 0, 2, 2, 1, 0, 5, 0, 0, 1,2,3,1,3,2, 5,1]
arr = np.asarray(list)
print(arr)
print(C)

avg = silhouette_score(array, arr)
print(avg)

