'''
Bianca Dizon
CSE 5523 HW 6
'''
import numpy as np
import argparse
import random 
from collections import deque as queue 

#invoke parser and parse through arguments
parser = argparse.ArgumentParser()
parser.add_argument('-data', action="store", dest="data_filename", type=str)
parser.add_argument('-k', action="store", dest="max_clusters", type=int)
parser.add_argument('-s', action="store", dest="max_clusterpts", type=int)
parser.add_argument('-output', action="store", dest="output_filename", type=str)
parser.add_argument('-d', action="store", dest="max_intra_dist", type=float)
args = parser.parse_args()

# read in data
data = np.genfromtxt(args.data_filename, delimiter=' ')
# print(data.size)
# print(data.shape)

# max number of iterations
MAX_ITER = 100
# K value=2
# K = 2
# euclidean distance between points
def euclidean(A, B):
    # print(np.linalg.norm(A-B))
    return np.linalg.norm(A-B)
# A = np.array([1,2,3,4])
# B = np.array([0,4,2,3])
# euclidean(A, B)

# calculates the intra-cluster distance of a given cluster
def icd(A,B):
    distance = euclidean(A,B)
    # print(distance * (1/len(A)))
    return distance * (1/len(A))
# A = np.array([1,2,3,4])
# B = np.array([0,4,2,3])
# icd(A, B)

# pick random centroids from the data points
def centroid(data, K):
    centroids = []
    centroid_ind = random.sample(range(0,data.shape[0]),K)    # random indices from data array for the 2 centroids
    for i in centroid_ind:
        centroids.append(data[i])   # centroids array contains the matrix of 2 centroid matrices
        # print(data[i])
    # print(centroids)
    # print("#####")
    # centroids = np.mat(np.zeros([num_centroids, data.shape[1]]))
    # print(centroids)
    # for i in range(data.shape[1]):
    #     min_pt = min(data[:,i])
    #     max_pt = max(data[:,i])
    #     range_pts = float(max_pt-min_pt)
    #     centroids[:,i] = min_pt + range_pts * np.random.rand(num_centroids,1)
    # print(centroids)
    return centroids

# kmeans algorithm
def kmeans(data, K):
    centroids = centroid(data, K)   # pick centroids
    centroids = np.array(centroids)     # change into an numpy array
    # print(centroids)
    # print(centroids[1])
    # print(centroids[:,0])   # column values at index 0
    # print(centroids[0,:])   #row values at index 0
    # cluster_info = np.mat(np.zeros([data.shape[0], 3]))
    cluster_info = []   # array containing info about cluster number, pt distance from centroid and point, points
    for i in range(data.shape[0]):  #creating the array of arrays
        new_list = []
        new_list.append([0])
        new_list.append([0.0]) #= [[0.0]] * 2
        new_list.append([0])
        cluster_info.append(new_list)
        # cluster_info.append([])
    cluster_info = np.array(cluster_info, dtype=object) #turn it into numpy array
    # # print(cluster_info)
    # cluster_info[0][2][0] = [1,2,3]
    # print(cluster_info[0])    
    # cluster_changed = True

    # find centroid for each data point 
    for i in range(data.shape[0]):
        min_dist = 100000
        min_index = -1
        for j in range(K):  # check which centroid data is closest to
            pt_dist = euclidean(centroids[j,:], data[i,:])
            if pt_dist < min_dist:
                min_dist = pt_dist
                min_index = j
        # if cluster_info[i][0][0] != min_index:
        #     cluster_changed = True
        cluster_info[i][0][0] = min_index   # add the cluster for point
        cluster_info[i][1][0] = min_dist    # distance from pt to cluster
        cluster_info[i][2][0] = data[i,:]   # data point array
    # for c in range(K):
    #     cluster_pts = data[np.nonzero(cluster_info[:,0,0].A == c)[0]]
    #     centroids[c,:] = np.mean(cluster_pts, axis=0)
    # print(len(cluster_info))
    # print(cluster_info)
    # print("&&&&")
    cluster_zero = []; cluster_one = []     # divide the two clusters
    for k in range(data.shape[0]):
        if cluster_info[k][0][0] == 0:
            cluster_zero.append(cluster_info[k][2][0]) 
        elif cluster_info[k][0][0] == 1:
            cluster_one.append(cluster_info[k][2][0])
    # print(len(cluster_one))
    # print(len(cluster_zero))
    # print(cluster_zero[0][2][0])
    # print(np.mean(cluster_zero, axis = 0))
    if K > 1:
        centroids[0, :] =  np.mean(cluster_zero, axis = 0)  # recalculate centroids
        centroids[1, :] = np.mean(cluster_one, axis = 0)
    elif K > 2:
        centroids[0, :] =  np.mean(cluster_zero, axis = 0)  # recalculate centroids

    return centroids, cluster_info
# kmeans(data, K)

# calculate best clusters for clustering
def best_clusters(data, K):
    min_sse = 10000000
     # find best clusters
    # print("@@@")
    num_iter = 100
    for i in range(num_iter):
        centroids, cluster_info = kmeans(data, K)
        sse = np.sum(cluster_info, axis=0)[1]   #sum the distance of all points from centroid
        # print(sse)
        # print(centroids)
        if sse < min_sse:   # if new cluster assignments are better then change the centroids and cluster_info
            min_sse = sse
            temp_centroids = centroids
            temp_cluster = cluster_info
    centroids = temp_centroids
    cluster_info = temp_cluster
    # separate clusters
    cluster_zero = []; cluster_one = [] # divide the points in each cluster
    for k in range(data.shape[0]):
        if cluster_info[k][0][0] == 0:
            cluster_zero.append(cluster_info[k][2][0]) 
        elif cluster_info[k][0][0] == 1:
            cluster_one.append(cluster_info[k][2][0])
    return centroids, cluster_zero, cluster_one
    # return centroids, cluster_info
# centroids, cluster_zero, cluster_one = best_clusters(data, 1)
# print(centroids)
# print(cluster_one)

# create class called Node as a data structure for the dendogram
# it is a binary tree
class Node: 
    
    def __init__(self, node_id, data): # will find nodes in binary tree using the node_id which is a random number assigned to a Node
        self.id = node_id   # use to find leaf Nodes later on
        self.data = data    # data will have all the pts associated with the node
        self.left = None
        self.right = None
  
# will return an array of all the leaf nodes node_id
def leaf_array(root: Node, leaf_nodes) -> None: 
    # i node is null, return 
    if (not root): 
        return
  
    # If node is leaf node,  
    # print its data 
    if (not root.left and 
        not root.right): 
        leaf_nodes.append(root.id)
        # print(root.id, end = " ") 
        return leaf_nodes
  
    # If left child exists,  
    # check for leaf recursively 
    if root.left: 
        leaf_array(root.left, leaf_nodes) 
  
    # If right child exists,  
    # check for leaf recursively 
    if root.right: 
        leaf_array(root.right, leaf_nodes) 
    return leaf_nodes

# print the node_ids of all the leaf nodes 
def printLeafNodes(root: Node) -> None: 
    # If node is null, return 
    if (not root): 
        return
  
    # If node is leaf node,  
    # print its data 
    if (not root.left and 
        not root.right): 
        # leaf_nodes.append(root.id)
        print(root.id,  
              end = " ") 
        return 
  
    # If left child exists,  
    # check for leaf recursively 
    if root.left: 
        printLeafNodes(root.left) 
  
    # If right child exists,  
    # check for leaf recursively 
    if root.right: 
        printLeafNodes(root.right) 
    # return leaf_nodes

# # Function to  print level order traversal of tree
# def printLevelOrder(root, all_nodes):
#     h = height(root)
#     for i in range(1, h+1):
#         printGivenLevel(root, i, all_nodes)
#     return all_nodes
 
 
# # Print nodes at a given level
# def printGivenLevel(root , level, all_nodes):
#     if root is None:
#         return
#     if level == 1:
#         # print(root.id,end=" ")
#         print(len(root.data),end=" ")
#         all_nodes.append(len(root.data))
#     elif level > 1 :
#         printGivenLevel(root.left , level-1, all_nodes)
#         printGivenLevel(root.right , level-1, all_nodes)
#     return all_nodes
# Function to do level order 
# traversal line by line 
# prints the structured dendogram in terminal
def levelOrder(root): 
      
    if (root == None): 
        return
  
    # Create an empty queue for 
    # level order tarversal 
    q = queue() 
  
    # To store front element of 
    # queue. 
    #node *curr 
  
    # Enqueue Root and None node. 
    q.append(root) 
    q.append(None) 
  
    while (len(q) > 1): 
        curr = q.popleft() 
        #q.pop() 
        # Condition to check occurrence of next level. 
        if (curr == None): 
           q.append(None) 
           print() 
  
        else: 
            # Pushing left child of current node. 
            if (curr.left): 
                q.append(curr.left) 
            # pushing right child of current node. 
            if (curr.right): 
                q.append(curr.right) 
  
            print(len(curr.data), end = " ") 

# find the height of the current node
def height(node):
    if node is None:
        return 0
    else :
        # Compute the height of each subtree 
        lheight = height(node.left)
        rheight = height(node.right)
 
        #Use the larger one
        if lheight > rheight :
            return lheight+1
        else:
            return rheight+1

# return array of leaf Nodes 
#returns the Node
def search(root: Node, node_id) -> None: 
  
    if root:
        # If node is null, return 
        if (root.id == node_id):
            # print("found")
            return root
        else:
            found_node = search(root.left, node_id)
            if found_node == None:
                found_node = search(root.right, node_id)
            return found_node
    else:
        return None

# will bisect the clusters and create the nodes in the binary tree of clusters (aka dendogram)
def bisecting(root, centroids, data, K, MAX_ITER, max_clusters, max_clusterpts, max_intra_dist, num_clusters, max_num_clusterpts):
    data = np.array(data)
    intra_dist = 0
    if root == None:    # return None if the root isn't created. should be created before using bisecting
        return
        # node_id = random.randint(0,100)
        # # print(node_id)
        # dendogram = Node(node_id,data)
        # num_clusters = 2
        # bisecting(dendogram, centroids, data, K, MAX_ITER, max_clusters, max_clusterpts, max_intra_dist, num_clusters, max_num_clusterpts)
    else:
        if  num_clusters < max_clusters and max_num_clusterpts < max_clusterpts and intra_dist < max_intra_dist:    # check if they are met
            if (max_clusters - num_clusters) % 2 != 0:
                centroids, cluster_zero, cluster_one = best_clusters(data,K)    # cluster the data using kmeans and finds best clusters
                num_clusters  = num_clusters + K    # update the number of clusters bc 2 were just created
                intra_zero = icd(centroids[0],cluster_zero) # find the intra-cluster distance of first cluster
                intra_one = icd(centroids[1],cluster_one) # find the intra-cluster distance of second cluster
                if intra_zero > intra_one:  # assign intra_dist according to biggest intra-cluster distance of both clusters
                    intra_dist = intra_zero
                else:
                    intra_dist = intra_one

                if max_num_clusterpts > len(cluster_zero):  # assigns the max number of clusterpoints to whichever cluster is bigger
                    max_num_clusterpts = len(cluster_zero)
                elif max_num_clusterpts >= len(cluster_one):
                    max_num_clusterpts = len(cluster_one)

                if len(cluster_zero) > len(cluster_one):    # if cluster zero is bigger then bisect further
                    node_id = random.randint(0,100)     # assign a node_id to new Node cluster
                    root.right = Node(node_id, cluster_zero)    # create new Node for cluster zero
                    # bisect cluster zero further and place it as the right child bc its bigger
                    bisecting(root.right,centroids, cluster_zero, K, MAX_ITER, max_clusters, max_clusterpts, max_intra_dist, num_clusters, max_num_clusterpts)
                    node_id = random.randint(0,100) # assign a node_id to new Node cluster
                    root.left = Node(node_id, cluster_one)  # create new Node for cluster one
                    # bisect cluster one further and place it as the left child bc its smaller
                    bisecting(root.left,centroids, cluster_one, K, MAX_ITER, max_clusters, max_clusterpts, max_intra_dist, num_clusters, max_num_clusterpts)
                elif len(cluster_one) >= len(cluster_zero):
                    node_id = random.randint(0,100) # assign a node_id to new Node cluster
                    root.right = Node(node_id, cluster_one) # create new Node for cluster one
                    # bisect cluster one further and place it as the right child bc its bigger
                    bisecting(root.right,centroids, cluster_one, K, MAX_ITER, max_clusters, max_clusterpts, max_intra_dist, num_clusters, max_num_clusterpts)
                    node_id = random.randint(0,100) # assign a node_id to new Node cluster
                    root.left = Node(node_id, cluster_zero) # create new Node for cluster zero
                    # bisect cluster zero further and place it as the left child bc its smaller
                    bisecting(root.left,centroids, cluster_zero, K, MAX_ITER, max_clusters, max_clusterpts, max_intra_dist, num_clusters, max_num_clusterpts)
            else:
                centroids, cluster_zero, cluster_one = best_clusters(data,1)    # cluster the data using kmeans and finds best clusters
                # cluster_zero = np.concatenate(cluster_zero, cluster_one)
                num_clusters  = num_clusters + 1    # update the number of clusters bc 2 were just created
                intra_dist = icd(centroids[0],cluster_zero) # find the intra-cluster distance
                max_num_clusterpts = len(cluster_zero)

                node_id = random.randint(0,100)     # assign a node_id to new Node cluster
                root.right = Node(node_id, cluster_zero)    # create new Node for cluster zero
                # bisect cluster zero further and place it as the right child bc its bigger
                bisecting(root.right,centroids, cluster_zero, K, MAX_ITER, max_clusters, max_clusterpts, max_intra_dist, num_clusters, max_num_clusterpts)
                node_id = random.randint(0,100) # assign a node_id to new Node cluster
                root.left = Node(node_id, cluster_one)  # create new Node for cluster one
                # bisect cluster one further and place it as the left child bc its smaller
        else:
            return(root)
        return(root)

    return

# print the cluster number for each data point in the data text file into the output text file
def print_clusters(dendogram, data, node_id_array, output_filename):
    file_out = open(output_filename, "w")   # open the output file 
    # node_1 = search(root,node_id_array[0])
    # node_2 = search(root,node_id_array[1])
    # node_3 = search(root,node_id_array[2])
    # node_4 = search(root,node_id_array[3])
    data = np.array(data)
    node_id_array = np.array(node_id_array)
    if len(data) != 0 and len(node_id_array) != 0:
        for i in range(len(data)):  # go over all the data points
            look = True # keep looking for cluster number if look = True
            j = 0   # iterate over all the leaf nodes
            while look and j < len(node_id_array):  # find the clustering number for ach data point
                node = search(root,node_id_array[j])
                if (data[i]==node.data).any():
                    file_out.write(str(j+1))    # print out cluster number
                    file_out.write("\n")    # print new line
                    look = False    # means stop looking, cluster number found
                j = j + 1   # iterate over leaf node array
            # look = True
    else:
        print("data array empty")


    file_out.close()    #close the output file
    return

if args.max_clusters > 1:   
    centroids = []  # empty centroid to send into bisecting
    node_id = random.randint(0,100) # node_id for the root of the dendogram
    # print(node_id)
    dendogram = Node(node_id,data)  # dendogram is the root of the dendogram
    # print(type(dendogram))
    num_clusters = 0
    max_num_clusterpts = 0
    # create dendogram and commence bisecting k-means
    K=2
    root = bisecting(dendogram, centroids, data, K, MAX_ITER, args.max_clusters, args.max_clusterpts, args.max_intra_dist, num_clusters, max_num_clusterpts)
    print("DENDOGRAM")
    levelOrder(root)    #print dendogram in stdout
    node_id_array = []  # array will hold the node_ids of all the leaf nodes
    node_id_array = leaf_array(root, node_id_array) # find all the leaf node ids and create an array



    # print cluster number of each datapoint into output file
    print_clusters(dendogram, data, node_id_array, args.output_filename)
    print()
    print("Number of clusters")
    # printLeafNodes(dendogram)
    print(len(node_id_array))
else:
    node_id = random.randint(0,100) # node_id for the root of the dendogram
    # print(node_id)
    dendogram = Node(node_id,data)  # dendogram is the root of the dendogram
    file_out = open(args.output_filename, "w")   # open the output file 
    file_out.write(str(1))    # print out cluster number
    file_out.write("\n")    # print new line
    file_out.close()    #close the output file

# max_clusters = 17
# num_clusters = 3
# if (max_clusters - num_clusters) % 2 != 0:
#     print("odd")
# else:
#     print("even")