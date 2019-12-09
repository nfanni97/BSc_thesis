import numpy as np

def file_len(fname):
    i = 0
    with open(fname,"r") as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def isMember(item,what):
    for w in what:
        if w==item:
            return True
    return False

def saveOutput(name,data,isLabel=False):
    if not isLabel:
        #savetxt can only handle 2-dim arrays so we have to reshape if we're not dealing with labels
        data = np.reshape(data,[data.shape[0],data.shape[1]*data.shape[2]])
        np.savetxt(name,data,fmt='%.6f',delimiter=' ',newline='\n')
    else:
        np.save(name,data)

path = "/home/nagfa5/raw_files/"
outpath = "/home/nagfa5/imaging/source/09/"
regions = ["middle","north","south"]
num_days =28 
period_start = [2160,4344,6552,8016]
period_stop = np.add(period_start,168)
num_train = 90
num_test = 20
bldgs = [5,7,9,12]
num_bldgs = len(bldgs)
num_regions = 3
num_classes = num_bldgs * num_regions

lengths = {
    "middle": 114,
    "north": 166,
    "south": 372
}

train_data = np.zeros((num_classes*num_train,num_days,24),dtype=np.single)
train_label = np.zeros((num_classes*num_train,num_classes))
test_data = np.zeros((num_classes*num_test,num_days,24),dtype=np.single)
test_label = np.zeros((num_classes*num_test,num_classes))
train_count = 0
test_count = 0

for region in regions:
    for c in range(num_bldgs):
        filename = region+"_"+str(bldgs[c])
        print(filename)
        filePath = path+filename+".csv"
        with open(filePath) as f:
            length = file_len(filePath)
            content = [None] * length
            for count, line in enumerate(f):
                content[count] = line.split(',')
                temp = []
                for i in range(len(period_start)):
                    temp = temp + content[count][period_start[i]:period_stop[i]]
                content[count] = temp
            content = np.array(content, dtype=np.double)
            #get train data and label:
            for i in range(num_train):
                rand = np.random.randint(low=0,high=length)
                temp = content[rand]
                temp = temp.reshape((num_days,24))
                train_data[train_count] = temp
                train_label[train_count] = np.zeros((1,num_classes))
                train_label[train_count,regions.index(region)*num_bldgs+c] = 1
                train_count += 1
            #get test data and label:
            for i in range(num_test):
                rand = np.random.randint(low=0, high=length)
                temp = content[rand]
                temp = temp.reshape((num_days, 24))
                test_data[test_count] = temp
                test_label[test_count] = np.zeros((1, num_classes))
                test_label[test_count, regions.index(region)*num_bldgs+c] = 1
                test_count += 1
#write data to file
saveOutput(outpath+"train_data",train_data)
saveOutput(outpath+"train_label",train_label,True)
saveOutput(outpath+"test_data",test_data)
saveOutput(outpath+"test_label",test_label,True)
print("hopefully done")
print(test_count)
print(train_count)
