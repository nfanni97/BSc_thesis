import numpy as np


def file_len(fname):
    i = 0
    with open(fname, "r") as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def saveOutput(name, data, isLabel=False):
    if not isLabel:
        #savetxt can only handle 2-dim arrays so we have to reshape if we're not dealing with labels
        data = np.reshape(data, [data.shape[0], data.shape[1]*data.shape[2]])
        np.savetxt(name, data, fmt='%.6f', delimiter=' ', newline='\n')
    else:
        np.save(name,data)
    return

path = "/home/nagfa5/GAN/04_groups/"
regions = ["middle", "north", "south"]
regionPath = "/home/nagfa5/raw_files/"
num_days = 7
period_start = np.arange(start=0,stop=363,step=7)
period_start *= 24
period_stop = period_start + num_days * 24
bldgs = [0]
# full service restaurant
num_regions = len(regions)
num_bldgs = len(bldgs)
num_classes = len(bldgs)

lengths = {
    "middle": 114,
    "north": 166,
    "south": 372
}
needed_length = lengths[min(lengths)]
num_samples = needed_length * period_start.shape[0]

data = np.zeros((3, num_classes * num_samples, num_days*24), dtype=np.single)
label = np.zeros((3, num_classes * num_samples, num_classes))
print(label.shape)

for r in range(num_regions):
    counter = 0
    for b in range(num_bldgs):
        fileName = regions[r]+"_"+str(bldgs[b])+".csv"
        filePath = regionPath+fileName
        print("working on: "+fileName)
        #read everything from file
        with open(filePath) as f:
            length = file_len(filePath)
            fileContent = [None] * length
            neededContent = np.zeros((length*52,num_days*24))
            for count, line in enumerate(f):
                fileContent[count] = line.split(',')
                for i in range(52):
                    neededContent[count*52+i] = fileContent[count][period_start[i]:period_stop[i]]
            neededContent = np.array(neededContent,dtype=np.single)
        #remove zero rows
        neededContent = neededContent[~np.all(neededContent<1e-6,axis=1)]
        #get num_samples non-zero sample out of each file
        for i in range(num_samples):
            label[r,counter,0] = 1
            data[r,counter,:] = neededContent[i,:]
            counter += 1
#write samples to file
for i in range(num_regions):
	np.save(path+"01_data_"+regions[i],data[i,:])
	np.save(path+"01_label_"+regions[i],label[i,:])
print(data.shape)
print(label.shape)
