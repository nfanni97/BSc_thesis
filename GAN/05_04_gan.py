import numpy as np
from datetime import datetime
import traceback
import sys #for command line arguments

from keras.models import Sequential,Model
from keras.layers import Dense,Conv2DTranspose,UpSampling2D,Conv2D
from keras.layers import BatchNormalization,Dropout,Activation,LeakyReLU
from keras.layers import Reshape,Flatten,Input
from keras.optimizers import Adam

class dcgan():
    def __init__(self,data,region,category):
        self.region = region
        self.category = category

        self.data = data
        self.img_shape = (168,168,1)
        self.noise_dim = 100

        #layer parameters
        self.momentum = 0.8
        self.alpha = 0.2
        self.learning_rate = 1e-4
        self.beta_1 = 0.5
        self.dropout = 0.4

        optimizer = Adam(lr=self.learning_rate,beta_1=self.beta_1)

        #discriminator
        self.discriminator = self.create_discriminator()
        self.discriminator.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
        #in the combined model, only the generator is trained
        self.discriminator.trainable = False

        #generator
        self.generator = self.create_generator()

        #GAN
        gan_input_noise = Input(shape=(self.noise_dim,))
        generated = self.generator(gan_input_noise)
        gan_output_validity = self.discriminator(generated)

        self.gan = Model(gan_input_noise,gan_output_validity)
        self.gan.compile(loss='binary_crossentropy',optimizer=optimizer)

    def create_discriminator(self):
        disc = Sequential()

        depth = 32
        # layers_1-3: feature extraction
        for i in range(5):
            if (i == 0):
                disc.add(Conv2D(depth * (2 ** i), input_shape=self.img_shape, kernel_size=3, strides=2, padding='same'))
            elif (i == 4):
                disc.add(Conv2D(depth * (2 ** i), kernel_size=3, strides=1, padding='same'))
            else:
                disc.add(Conv2D(depth * (2 ** i), kernel_size=3, strides=2, padding='same'))
            disc.add(LeakyReLU(alpha=self.alpha))
            disc.add(Dropout(self.dropout))
        # layer_4: get the probability
        disc.add(Flatten())
        disc.add(Dense(1))
        disc.add(Activation('sigmoid'))

        img = Input(shape=self.img_shape)
        validity = disc(img)
        return Model(img,validity)

    def create_generator(self):
        gen = Sequential()

        # layer_1: (100,1) -> (21,21,256)
        dim_1_2 = 21
        filter_size = 256
        gen.add(Dense(dim_1_2 * dim_1_2 * filter_size, input_dim=self.noise_dim))
        gen.add(BatchNormalization(momentum=self.momentum))
        gen.add(Activation('relu'))
        gen.add(Reshape((dim_1_2, dim_1_2, filter_size)))
        gen.add(Dropout(self.dropout))
        # layers_2-4: first 2 dimensions multiplied by 2, third dimension divided by 2
        for i in range(3):
            dim_1_2 *= 2
            filter_size /= 2
            gen.add(UpSampling2D())
            gen.add(Conv2DTranspose(filters=int(filter_size), kernel_size=3, padding='same'))
            gen.add(BatchNormalization(momentum=self.momentum))
            gen.add(Activation('relu'))
        # layer_6: (168,168,16) -> (168,168,1) image
        gen.add(Conv2DTranspose(1, 5, padding='same'))
        gen.add(Activation('tanh'))

        noise = Input(shape=(self.noise_dim,))
        img = gen(noise)
        return Model(noise,img)

    def train(self,steps,batch_size,save_interval,path,num_img):
        #labels
        real_label = np.ones((batch_size,1))
        fake_label = np.zeros((batch_size,1))

        #graph data
        d_loss = np.zeros((steps,1))
        d_acc = np.zeros((steps,1))
        g_loss = np.zeros((steps,1))

        #other regions
        other_1 = np.load(path + '03_img_' + self.category + '_' + sys.argv[2] + '.npy')
        other_2 = np.load(path + '03_img_' + self.category + '_' + sys.argv[3] + '.npy')

        try:
            for i in range(steps):
                #train discriminator
                idx = np.random.randint(low=0,high=self.data.shape[0],size=batch_size)
                real_img = self.data[idx]
                real_img = real_img.reshape([batch_size]+list(self.img_shape))
                noise = np.random.normal(loc=0,scale=1,size=(batch_size,self.noise_dim))
                fake_img = self.generator.predict(noise)
                #other regions as fake data
                other_idx = np.random.randint(low=0,high=other_1.shape[0],size=batch_size//2)
                other_1_img = other_1[other_idx]
                other_1_img = other_1_img.reshape([batch_size//2]+list(self.img_shape))
                other_2_img = other_2[other_idx]
                other_2_img = other_2_img.reshape([batch_size//2]+list(self.img_shape))
                other_img = np.concatenate((other_1_img,other_2_img),axis=0)

                d_loss_real = self.discriminator.train_on_batch(real_img,real_label)
                d_loss_fake_gen = self.discriminator.train_on_batch(fake_img,fake_label)
                d_loss_fake_other = self.discriminator.train_on_batch(other_img,fake_label)
                d_loss_fake = 0.5 * np.add(d_loss_fake_other,d_loss_fake_gen)
                (d_loss[i],d_acc[i]) = 0.5 * np.add(d_loss_real,d_loss_fake)

                #train generator
                g_loss[i] = self.gan.train_on_batch(noise,real_label)

                #log
                log_msg = "%d. step:\tgenerator loss: %f\tdiscriminator loss: %f\tdiscriminator acc: %f" % (
                i, g_loss[i], d_loss[i], d_acc[i])
                print(log_msg)
                if (i + 1) % save_interval == 0:
                    self.save_everything(i + 1, path, d_loss, d_acc, g_loss, num_img)

        except KeyboardInterrupt as k:
          with open(path + self.category + '_stopped', 'w') as f:
            f.write('Script was stopped on the %d. step at %s' % (i, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
          self.save_everything(i, path, d_loss, d_acc, g_loss, num_img)

        except Exception as e:
            print('exception')
            with open(path + self.category + '_exception', 'w') as f:
              f.write('Script was stopped on the %d. step at %s' % (i, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
              f.write('\n')
              traceback.print_exc(file=f)
            self.save_everything(i, path, d_loss, d_acc, g_loss, num_img)

    def save_everything(self, i, path, d_loss, d_acc, g_loss, num_img):
      self.save_results(i, path, d_loss, d_acc, g_loss)
      self.save_models(i, path)
      self.save_images(i, path, num_img)

    def save_results(self, i, path, d_loss, d_acc, g_loss):
      np.savetxt(path + self.category + '_d_loss', d_loss, delimiter='\n')
      np.savetxt(path + self.category + '_d_acc', d_acc, delimiter='\n')
      np.savetxt(path + self.category + '_g_loss', g_loss, delimiter='\n')

    def save_models(self, i, path):
      self.gan.save(path + 'models/' + self.category + '/gan_%d' % i)
      self.discriminator.save(path + 'models/' + self.category + '/disc_%d' % i)
      self.generator.save(path + 'models/' + self.category + '/gen_%d' % i)

    def save_images(self, i, path, img_num):
      noise = np.random.normal(0, 1, (img_num, self.noise_dim))
      generated = self.generator.predict(noise)
      generated = np.array(generated, dtype=np.float32)
      generated = generated.reshape([img_num, self.img_shape[0],self.img_shape[1]])
      np.save(path + 'generated/' + self.category + '/%d' % i, generated)

def load_data(path,fileName):
    return np.load(path+fileName)

# inPath = '/home/fanni/Dokumentumok/PPKE-ITK/7.felev/Szakdoga/GAN/03/'
# dataFile = 'reduced_data.npy'
path = '/home/nagfa5/GAN/05_stocha_b/'
region = sys.argv[1]
category = sys.argv[4]
dataFile = '03_img_'+sys.argv[4]+'_'+region+'.npy'
batch_size = 50
save_interval = 50
steps = 2000
num_img = 4

data = load_data(path,dataFile)
GAN = dcgan(data,region,category)
GAN.train(steps,batch_size,save_interval,path,num_img)
