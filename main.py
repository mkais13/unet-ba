from model import *
from data import *
import os
import argparse

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#testcomment 

parser = argparse.ArgumentParser(description='set Hyperparameters for training')
parser.add_argument('-e' , '--epochs', type=int, metavar='epochs', nargs='?', default=5, const=5, help='Number of Epochs')
#parser.add_argument('-s' , '--steps', type=int, metavar='steps', nargs='?', default=300, const=300, help='Number of Steps per Epoch')
parser.add_argument('-bs' , '--batchsize', type=int, metavar='batchsize',nargs='?', default=2, const=2, help='Batch Size')
parser.add_argument('-lf' , '--lossfunction', metavar='lossfunction',nargs='?', default='binary_crossentropy', const='binary_crossentropy', help='loss function for the Model')
parser.add_argument('-opt' , '--optimizer', metavar='optimizer',nargs='?', default="Adam", const="Adam", help='optimizer function for the model')
parser.add_argument('-lr' , '--learningrate' , type=float, metavar='learningrate',nargs='?', default= 1e-4, const= 1e-4, help='learning rate for the model')
args = parser.parse_args()

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')


myGene = trainGenerator(args.batchsize,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

model = unet(args.lossfunction, args.optimizer, args.learningrate)
dirpath = '/scratch/tmp/m_kais13/checkpoints'
os.makedirs(dirpath, exist_ok=True)
filename = 'bs{0}-lf{1}-opt{2}-lr{3}.h5'.format(args.batchsize,args.lossfunction,args.optimizer,args.learningrate)
cb_checkpointer = ModelCheckpoint(filepath = os.path.join(dirpath, filename), monitor = 'loss', save_best_only = False, mode = 'auto', verbose=1)
#model_checkpoint = ModelCheckpoint("/scratch/tmp/m_kais13/checkpoints/unetmembranetest.h5", monitor='loss',verbose=1, save_best_only=False)
num_images = 30
model.fit_generator(myGene,steps_per_epoch=(args.batchsize/num_images),epochs=args.epochs,callbacks=[cb_checkpointer])
#model.fit_generator(myGene,steps_per_epoch=args.steps,epochs=args.epochs)
#model.save("/scratch/tmp/m_kais13/checkpoints/unetmembranetest")

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene,30,verbose=1)
#saveResult("data/membrane/test",results)