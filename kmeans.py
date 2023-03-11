import numpy as np

def kmeans(feats, k, max_iter=1000):
    #init
    center_idx = np.random.choice(feats.shape[0], k)
    centers = feats[center_idx,:]
    yprev = None
    for i in range(max_iter):
        #compute distances
        squared_distance = np.sum(feats*feats, axis=1, keepdims=True) + np.sum(centers*centers, axis=1, keepdims=True).T - 2*np.matmul(feats, centers.T)
        #update assignments
        y = np.argmin(squared_distance, axis=1)
        #squared distance of each point to its center
        d = np.min(squared_distance, axis=1)
        #print objective: sum of squared distance of each point to its assigned center
        obj = np.sum(d)
        print('Iteration {:d}: {:f}'.format(i, obj))
        if np.all(y==yprev):
            break


        #update centers
        for j in range(k):
            if np.sum(y==j) == 0:
                #no data points assigned to the j-th cluster
                #assign data point farthest from its assigned cluster
                idx = np.argmin(d)
                d[idx]=0
                y[idx]=j

            #new center is mean of all assigned datapoints
            centers[j,:] = np.mean(feats[y==j,:], axis=0)
        
        yprev=y
    return y


def kmeans_with_color(img, k):
  pass

def kmeans_with_color_posn(img, k, alpha):
  pass