from model import *
from data import *
import os
import math
import argparse

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#testcomment 

parser = argparse.ArgumentParser(description='set Hyperparameters for training')
parser.add_argument('-e' , '--epochs', type=int, metavar='epochs', nargs='?', default=3, const=1, help='Number of Epochs')
#parser.add_argument('-s' , '--steps', type=int, metavar='steps', nargs='?', default=300, const=300, help='Number of Steps per Epoch')
parser.add_argument('-bs' , '--batchsize', type=int, metavar='batchsize',nargs='?', default=3, const=3, help='Batch Size')
parser.add_argument('-lf' , '--lossfunction', metavar='lossfunction',nargs='?', default='binary_crossentropy', const='binary_crossentrop', help='loss function for the Model')
parser.add_argument('-ki' , '--kernelinitializer', metavar='kernelinitializer',nargs='?', default='he_normal', const='he_normal', help='kernel initializer for the Model')
parser.add_argument('-opt' , '--optimizer', metavar='optimizer',nargs='?', default="Adam", const="Adam", help='optimizer function for the model')
#parser.add_argument('-lr' , '--learningrate' , type=float, metavar='learningrate',nargs='?', default= 1e-4, const= 1e-4, help='learning rate for the model')
parser.add_argument('-tf' , '--topologyfactor', type=float, metavar='topologyfactor',nargs='?', default=1.0, const=1, help='')
args = parser.parse_args()

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
                
num_images = 30

myGene = trainGenerator(args.batchsize,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
run_identifier = 'bs{0}-lf{1}-opt{2}-tf{3}-ki{4}'.format(args.batchsize,args.lossfunction,args.optimizer, args.topologyfactor, args.kernelinitializer)
model = unet(args.lossfunction, args.optimizer, args.topologyfactor, args.kernelinitializer)
checkpointpath = '/scratch/tmp/m_kais13/run4/checkpoints/' + run_identifier
tensorboardpath = '/scratch/tmp/m_kais13/run4/losslogs'
resultpath = '/scratch/tmp/m_kais13/run4/results/' + run_identifier
os.makedirs(tensorboardpath, exist_ok=True)
os.makedirs(checkpointpath, exist_ok=True)
os.makedirs(resultpath, exist_ok=True)


validGene = validGenerator(args.batchsize, "data/membrane/train", "image", "label")
validGene2 = validGenerator(args.batchsize, "data/membrane/train", "image", "label")

class TensorBoardWrapper(TensorBoard):

    def __init__(self, batch_gen, nb_steps, b_size, **kwargs):
        super(TensorBoardWrapper, self).__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps   # Number of times to call next() on the generator.
        self.batch_size = b_size

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib, tb = next(self.batch_gen)
            if imgs is None and tags is None:
                imgs = np.zeros(((self.nb_steps * self.batch_size,) + ib.shape[1:]), dtype=np.float32)
                tags = np.zeros(((self.nb_steps * self.batch_size,) + tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
              
        return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)








cb_tensorboard = TensorBoardWrapper(validGene2,math.ceil(num_images/args.batchsize), args.batchsize, log_dir= os.path.join(tensorboardpath, run_identifier), histogram_freq=1, batch_size=args.batchsize)

#cb_tensorboard = TensorBoard(log_dir= os.path.join(tensorboardpath, run_identifier), histogram_freq=1,write_grads=True, write_graph=True)
cb_checkpointer = ModelCheckpoint(filepath = os.path.join(checkpointpath,run_identifier+"-e{epoch}.h5"), monitor = 'loss', mode = 'auto', verbose=1)
#model_checkpoint = ModelCheckpoint("/scratch/tmp/m_kais13/checkpoints/unetmembranetest.h5", monitor='loss',verbose=1, save_best_only=False)

model.fit_generator(myGene,(num_images/args.batchsize)*10,epochs=args.epochs,callbacks=[cb_checkpointer, cb_tensorboard] , validation_data=validGene, validation_steps=3)
#model.fit_generator(myGene,steps_per_epoch=args.steps,epochs=args.epochs)
#model.save("/scratch/tmp/m_kais13/checkpoints/unetmembranetest")

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene,30,verbose=1)
saveResult(resultpath,results)