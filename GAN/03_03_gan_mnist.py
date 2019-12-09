import numpy as np
from datetime import datetime
import traceback

from keras.models import Sequential,Model
from keras.layers import Dense,Conv2DTranspose,UpSampling2D,Conv2D
from keras.layers import BatchNormalization,Dropout,Activation,LeakyReLU
from keras.layers import Reshape,Flatten,Input,ZeroPadding2D
from keras.optimizers import Adam

from keras.datasets import mnist

class dcgan():
    def __init__(self,data):
        self.data = data
        self.img_shape = (28,28,1)
        self.noise_dim = 100

        #other parameters
        self.momentum = 0.8
        self.alpha = 0.2
        self.learning_rate = 1e-4

        optimizer = Adam(lr=self.learning_rate,beta_1=0.5)

        #discriminator
        self.discriminator = self.create_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        #in the combined model, only the generator is trained
        self.discriminator.trainable = False

        #generator
        self.generator = self.create_generator()

        #combined model
        gan_input_noise = Input(shape=(self.noise_dim,))
        generated = self.generator(gan_input_noise)
        gan_output_validity = self.discriminator(generated)
        self.gan = Model(gan_input_noise,gan_output_validity)
        self.gan.compile(loss='binary_crossentropy',optimizer=optimizer)

    def create_discriminator(self):
        disc = Sequential()

        disc.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        disc.add(LeakyReLU(alpha=self.alpha))
        disc.add(Dropout(0.25))
        disc.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        disc.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        disc.add(BatchNormalization(momentum=self.momentum))
        disc.add(LeakyReLU(alpha=self.alpha))
        disc.add(Dropout(0.25))
        disc.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        disc.add(BatchNormalization(momentum=self.momentum))
        disc.add(LeakyReLU(alpha=self.alpha))
        disc.add(Dropout(0.25))
        disc.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        disc.add(BatchNormalization(momentum=self.momentum))
        disc.add(LeakyReLU(alpha=self.alpha))
        disc.add(Dropout(0.25))
        disc.add(Flatten())
        disc.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        validity = disc(img)

        return Model(img, validity)

    def create_generator(self):
        gen = Sequential()

        gen.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.noise_dim))
        gen.add(Reshape((7, 7, 128)))
        gen.add(UpSampling2D())
        gen.add(Conv2D(128, kernel_size=3, padding="same"))
        gen.add(BatchNormalization(momentum=self.momentum))
        gen.add(Activation("relu"))
        gen.add(UpSampling2D())
        gen.add(Conv2D(64, kernel_size=3, padding="same"))
        gen.add(BatchNormalization(momentum=self.momentum))
        gen.add(Activation("relu"))
        gen.add(Conv2D(1, kernel_size=3, padding="same"))
        gen.add(Activation("tanh"))

        noise = Input(shape=(self.noise_dim,))
        img = gen(noise)

        return Model(noise, img)

    def train(self,path,steps,save_interval,save_gen_img,batch_size):
        #labels
        real_label = np.ones((batch_size,1))
        fake_label = np.zeros((batch_size,1))

        #losses
        d_loss = np.zeros((steps,1))
        d_acc = np.zeros((steps,1))
        g_loss = np.zeros((steps,1))

        try:
            for i in range(steps):
                #train discriminator
                idx = np.random.randint(0,self.data.shape[0],batch_size)
                real_img = self.data[idx]
                real_img = real_img.reshape([batch_size]+list(self.img_shape))
                noise = np.random.normal(0,1,(batch_size,self.noise_dim))
                fake_img = self.generator.predict(noise)

                d_loss_real = self.discriminator.train_on_batch(real_img,real_label)
                d_loss_fake =  self.discriminator.train_on_batch(fake_img,fake_label)
                (d_loss[i], d_acc[i]) = 0.5 * np.add(d_loss_fake, d_loss_real)

                #train generator
                g_loss[i] = self.gan.train_on_batch(noise,real_label)

                #log
                log_msg = "%d. step:\tgenerator loss: %f\tdiscriminator loss: %f\tdiscriminator acc: %f" % (i,g_loss[i],d_loss[i],d_acc[i])
                print(log_msg)
                if (i+1)%save_interval == 0:
                    self.save_everything(i+1,path,d_loss,d_acc,g_loss,save_gen_img)
        except KeyboardInterrupt as k:
            with open(path+'stopped','w') as f:
                f.write('Script was stopped on the %d. step at %s' % (i,datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            self.save_everything(i,path,d_loss,d_acc,g_loss,save_gen_img)
        except Exception as e:
            with open(path+'exception', 'w') as f:
                f.write('Script was stopped on the %d. step at %s' % (i,datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                f.write()
                traceback.print_exc(file=f)
            self.save_everything(i,path,d_loss,d_acc,g_loss,save_gen_img)

    def save_everything(self,i,path,d_loss,d_acc,g_loss,save_gen_img):
        self.save_results(i, path, d_loss, d_acc, g_loss)
        self.save_models(i, path)
        self.save_images(i, path, save_gen_img)

    def save_results(self,i,path,d_loss,d_acc,g_loss):
        np.savetxt(path+'d_loss',d_loss,delimiter='\n')
        np.savetxt(path + 'd_acc', d_acc, delimiter='\n')
        np.savetxt(path + 'g_loss', g_loss, delimiter='\n')

    def save_models(self,i,path):
        self.gan.save(path+'models/gan_%d' % i)
        self.discriminator.save(path+'models/disc_%d' % i)
        self.generator.save(path+'models/gen_%d' % i)

    def save_images(self,i,path,img_num):
        noise = np.random.normal(0,1,(img_num,self.noise_dim))
        generated = self.generator.predict(noise)
        generated = np.array(generated,dtype=np.float32)
        generated = generated.reshape([img_num,28,28])
        np.save(path+'generated/%d' % i,generated)

def load_data():
    (x_train, _), (_, _) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    # convert shape of x_train from (60000, 28, 28) to (60000, 784)
    # 784 columns per row
    x_train = x_train.reshape(60000, 784)
    return x_train

path = '/home/fanni/Dokumentumok/PPKE-ITK/7.felev/Szakdoga/GAN/03_a/'
# dataFile = 'reduced_data.npy'
# inPath = '/home/nagfa5/GAN/02_a/'
# outPath = '/home/nagfa5/GAN/03_a/'
# dataFile = '01_data_without_PAA.npy'
batch_size = 50
steps = 2000
save_interval = 10
gen_img_num = 4

data = load_data()
GAN = dcgan(data)
GAN.generator.summary()
print()
GAN.discriminator.summary()
#GAN.train(path,steps,save_interval,gen_img_num,batch_size)
