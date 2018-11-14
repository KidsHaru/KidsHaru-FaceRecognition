from sklearn.cluster import DBSCAN
import numpy as np

def cluster(data, indexE, encoding):
    clt = DBSCAN(eps=0.22, min_samples=2, metric="euclidean")
    X = clt.fit(encoding)
    # print(X)

    # label 결정
    label_ids = np.unique(clt.labels_)
    num_unique_faces = len(np.where(label_ids > -1)[0])
    print("clustered %d unique faces." % num_unique_faces)
    print(len(label_ids))
    print("===============")

    for n in range(len(data)):
            data.loc[data.index == n, 'clustering'] = "empty"
        
    count = 0
    final = 0
    for label_id in label_ids:
        dir_name = "ID%d" % label_id
        print(dir_name)

        # find all indexes of label_id
        indexes = np.where(clt.labels_ == label_id)[0]
        print(indexes)
        for n in range(len(indexE)):
                len_t = int(indexE[n].split(".")[0])
                
                result = data.loc[data.index == len_t]['clustering']
                if result.item() == "empty":
                        result = ""
                        temp = result + str(label_id) + "."
                        data.loc[data.index == int(indexE[n].split(".")[0]), 'clustering'] = temp
                else:
                        temp = result + str(label_id) + "."
                        data.loc[data.index == int(indexE[n].split(".")[0]), 'clustering'] = temp

        if label_id > -1:
                count += len(indexes)
                final += len(indexes)
        else:
                final += len(indexes)
    
    print(count, final, " ", round(count / final * 100, 2), "%")
        
