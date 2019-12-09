import numpy as np
import matplotlib.pyplot as plt

def file_len(fname):
    i = 0
    with open(fname,"r") as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def saveOutput(name,data,isLabel=False):
    if not isLabel:
        #savetxt can only handle 2-dim arrays so we have to reshape if we're not dealing with labels
        data = np.reshape(data,[data.shape[0],data.shape[1]*data.shape[2]])
        np.savetxt(name,data,fmt='%.6f',delimiter=' ',newline='\n')
    else:
        np.save(name,data)
    return

path = "/home/fanni/Dokumentumok/PPKE-ITK/7.felev/Szakdoga/GAN/"
regions = ["middle", "north", "south"]
num_days = 7
period_start = 2160
period_stop = period_start + num_days * 24
bldgs = [0]
# full service restaurant
num_regions = len(regions)
num_bldgs = len(bldgs)
num_classes = num_bldgs * num_regions

lengths = {
    "middle": 114,
    "north": 166,
    "south": 372
}
num_samples = lengths[min(lengths)]

data = np.zeros((num_classes * num_samples, num_days*24), dtype=np.single)
label = np.zeros((num_classes * num_samples, num_classes))

counter = 0

for r in range(num_regions):
    for b in range(num_bldgs):
        fileName = regions[r]+"_"+str(bldgs[b])+".csv"
        filePath = path+fileName
        print("working on: "+fileName)
        #read everything from file
        with open(filePath) as f:
            length = file_len(filePath)
            fileContent = [None] * length
            for count, line in enumerate(f):
                fileContent[count] = line.split(',')
                fileContent[count] = fileContent[count][period_start:period_stop]
            fileContent = np.array(fileContent,dtype=np.single)
        #remove zero rows
        fileContent = fileContent[~np.all(fileContent<1e-6,axis=1)]
        #get num_samples non-zero sample out of each file
        for i in range(num_samples):
            label[counter,r*num_bldgs+b] = 1
            data[counter,:] = fileContent[i,:]
            counter += 1
#write samples to file
np.save(path+"data",data)
np.save(path+"label",label)
print(data)
plt.plot(data[-1,:])