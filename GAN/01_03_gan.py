import numpy as np

from keras.layers import Dense,Activation,Flatten,Reshape,Input
from keras.layers import Conv2D,Conv2DTranspose,UpSampling2D
from keras.layers import LeakyReLU,Dropout
from keras.layers import BatchNormalization
from keras.models import Sequential,load_model
from keras.optimizers import Adam

##############################
##########FUNCTIONS###########
##############################

def adam_optimizer(learning_rate=0.0002,beta_1=0.5):
    return Adam(lr=learning_rate,beta_1=beta_1)

class DCGAN:
    def _init_(self):
        self.D = None
        self.DM = None
        self.G = None
        self.GAN = None
        self.x = None
        self.y = None

    def load_data(self, path_data, path_label):
        self.x = np.load(path_data)
        self.y = np.load(path_label)

    def save_images(self,filepath,step,num=4):
        dim = int(num**0.5)
        noise = np.random.uniform(-1,1,size=[num,100])
        generated = self.G.predict(noise)
        generated = np.array(generated,dtype=np.float32)
        generated.reshape([num,168,168])
        np.save(filepath+"generated/%d" % step,generated)

    #generator: generates 168x168 images from 100x1 random noise (between -1 and 1)
    def create_generator(self,dropout=0.4,momentum=0.9):
        self.G = Sequential()
        #layer_1: (100,1) -> (21,21,256)
        dim_1_2 = 21
        filter_size = 256
        self.G.add(Dense(dim_1_2*dim_1_2*filter_size,input_dim=100))
        self.G.add(BatchNormalization(momentum=momentum))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim_1_2,dim_1_2,filter_size)))
        self.G.add(Dropout(dropout))
        #layers_2-4: first 2 dimensions multiplied by 2, third dimension divided by 2
        for i in range(3):
            dim_1_2 *= 2
            filter_size /= 2
            self.G.add(UpSampling2D())
            self.G.add(Conv2DTranspose(filters=int(filter_size),kernel_size=3,padding='same'))
            self.G.add(BatchNormalization(momentum=momentum))
            self.G.add(Activation('relu'))
        #layer_6: (168,168,16) -> (168,168,1) image
        self.G.add(Conv2DTranspose(1,5,padding='same'))
        self.G.add(Activation('sigmoid'))
        return self.G

    #discriminator: from a (168,168,3) image it tells how real it is on a scale of 0 to 1
    def create_discriminator(self,dropout=0.4,alpha=0.2):
        self.D = Sequential()
        depth = 32
        input_shape = (168,168,1)
        #layers_1-3: feature extraction
        for i in range(5):
            if(i==0):
                self.D.add(Conv2D(depth * (2 ** i),input_shape=input_shape,kernel_size=3,strides=2,padding='same'))
            elif(i==4):
                self.D.add(Conv2D(depth * (2 ** i), kernel_size=3, strides=1, padding='same'))
            else:
                self.D.add(Conv2D(depth * (2 ** i), kernel_size=3, strides=2, padding='same'))
            self.D.add(LeakyReLU(alpha=alpha))
            self.D.add(Dropout(dropout))
        #layer_4: get the probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        return self.D

    def create_discriminator_model(self,optimizer):
        self.DM = Sequential()
        self.DM.add(self.D)
        self.DM.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        return self.DM

    def compile_gan(self,optimizer):
        self.GAN = Sequential()
        self.GAN.add(self.G)
        self.GAN.add(self.D)
        self.GAN.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        return self.GAN

    def train(self,filepath,steps=2000,batch_size=10,save_interval=1):
        noise = None
        A_loss = np.zeros([steps,1])
        D_loss = np.zeros([steps, 1])
        A_acc = np.zeros([steps, 1])
        D_acc = np.zeros([steps, 1])
        for i in range(steps):
            real = self.x[np.random.randint(low=0,high=self.x.shape[0],size=batch_size)]
            real = np.expand_dims(real, axis=-1)
            noise = np.random.uniform(-1,1,size=[batch_size,100])
            fake = self.G.predict(noise)
            x = np.concatenate([real, fake])
            y = np.zeros([2*batch_size,1])
            y[:batch_size] = 1
            d_loss = self.DM.train_on_batch(x,y)

            y = np.ones([batch_size,1])
            noise = np.random.uniform(-1,1,size=[batch_size,100])
            a_loss = self.GAN.train_on_batch(noise,y)
            log_msg = "%d. step:  [D loss: %f, D acc: %f]\t[A loss: %f, A acc: %f]" % (i,d_loss[0],d_loss[1],a_loss[0],a_loss[1])
            print(log_msg)
            [A_loss[i], A_acc[i]] = a_loss[:2]
            [D_loss[i], D_acc[i]] = d_loss[:2]
            if(save_interval>0 and i%save_interval==0):
                np.savetxt(filepath+'d_loss',D_loss,delimiter='\n')
                np.savetxt(filepath + 'a_loss', A_loss,delimiter='\n')
                np.savetxt(filepath + 'd_acc', D_acc,delimiter='\n')
                np.savetxt(filepath + 'a_acc', A_acc,delimiter='\n')
                self.save_images(filepath, i, 4)
                self.GAN.save(filepath+"models/gan_%d.h"%i)

##############################
##########PARAMETERS##########
##############################

path = "/home/nagfa5/GAN/01/"
data_file = "01_data_gasf_without_PAA.npy"
label_file = "01_label.npy"
learning_rate = 0.0002
beta_1 = 0.5
dropout = 0.4
momentum = 0.9#for batch normalization
alpha = 0.2#for leaky relu

##############################
#############CODE#############
##############################

gan = DCGAN()
gan.load_data(path+data_file,path+label_file)
gan.create_discriminator(dropout,alpha)
#gan.D.summary()
gan.create_generator(dropout,momentum)
#gan.G.summary()
gan.create_discriminator_model(optimizer=adam_optimizer(learning_rate,beta_1))
gan.compile_gan(optimizer=adam_optimizer(learning_rate,beta_1))
#gan.GAN.summary()
gan.train(path,steps=1500,save_interval=100)
