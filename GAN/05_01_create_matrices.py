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

path = "/home/nagfa5/GAN/05_stocha_c/"
regions = ["middle", "north", "south"]
regionPath = "/home/nagfa5/raw_files/"
num_days = 7
period_start = np.arange(start=0,stop=363,step=7)
period_start *= 24
period_stop = period_start + num_days * 24
bldgs = [0,2,7]
# full service restaurant, large hotel, primary school, respectively
num_regions = len(regions)
num_bldgs = len(bldgs)

lengths = {
    "middle": 114,
    "north": 166,
    "south": 372
}
needed_length = lengths[min(lengths)]
num_samples = needed_length * period_start.shape[0]

# region, building, samples, hours
data = np.zeros((3, 3,num_samples, num_days*24), dtype=np.single)
for r in range(num_regions):
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
            data[r,b,i,:] = neededContent[i,:]
#write samples to file
bldgs_map = {
    0: 'restaurant',
    2: 'hotel',
    7: 'school'
}
for i in range(num_regions):
    for b in range(num_bldgs):
        np.save(path+bldgs_map[bldgs[b]]+"/01_data_"+regions[i]+'_'+bldgs_map[bldgs[b]],data[i,b,:])
