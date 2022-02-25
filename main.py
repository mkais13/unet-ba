from model import *
from data import *
import os
import argparse

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#testcomment 

parser = argparse.ArgumentParser(description='set Hyperparameters for training')
parser.add_argument('-e' , '--epochs', type=int, metavar='epochs', nargs='?', default=1, const=1, help='Number of Epochs')
#parser.add_argument('-s' , '--steps', type=int, metavar='steps', nargs='?', default=300, const=300, help='Number of Steps per Epoch')
parser.add_argument('-bs' , '--batchsize', type=int, metavar='batchsize',nargs='?', default=2, const=2, help='Batch Size')
parser.add_argument('-lf' , '--lossfunction', metavar='lossfunction',nargs='?', default='binary_crossentropy', const='binary_crossentropy', help='loss function for the Model')
parser.add_argument('-ki' , '--kernelinitializer', metavar='kernelinitializer',nargs='?', default='he_normal', const='he_normal', help='kernel initializer for the Model')
parser.add_argument('-opt' , '--optimizer', metavar='optimizer',nargs='?', default="Adam", const="Adam", help='optimizer function for the model')
#parser.add_argument('-lr' , '--learningrate' , type=float, metavar='learningrate',nargs='?', default= 1e-4, const= 1e-4, help='learning rate for the model')
parser.add_argument('-tf' , '--topologyfactor', type=float, metavar='topologyfactor',nargs='?', default=2, const=1, help='')
args = parser.parse_args()

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
                


myGene = trainGenerator(args.batchsize,'data/membrane/train','image','label',dict(),save_to_dir = None)
run_identifier = 'bs{0}-lf{1}-opt{2}-tf{3}-ki{4}'.format(args.batchsize,args.lossfunction,args.optimizer, args.topologyfactor, args.kernelinitializer)
model = unet(args.lossfunction, args.optimizer, args.topologyfactor, args.kernelinitializer)
checkpointpath = '/scratch/tmp/m_kais13/checkpoints'
tensorboardpath = '/scratch/tmp/m_kais13/losslogs'
resultpath = '/scratch/tmp/m_kais13/results/' + run_identifier
os.makedirs(tensorboardpath, exist_ok=True)
os.makedirs(checkpointpath, exist_ok=True)
os.makedirs(resultpath, exist_ok=True)
cb_tensorboard = TensorBoard(log_dir= os.path.join(tensorboardpath, run_identifier))
cb_checkpointer = ModelCheckpoint(filepath = os.path.join(checkpointpath,run_identifier+"-e{epoch}.h5"), monitor = 'loss', mode = 'auto', verbose=1)
#model_checkpoint = ModelCheckpoint("/scratch/tmp/m_kais13/checkpoints/unetmembranetest.h5", monitor='loss',verbose=1, save_best_only=False)
num_images = 30
model.fit_generator(myGene,steps_per_epoch=(num_images/args.batchsize),epochs=args.epochs,callbacks=[cb_checkpointer, cb_tensorboard])

#model.fit_generator(myGene,steps_per_epoch=args.steps,epochs=args.epochs)
#model.save("/scratch/tmp/m_kais13/checkpoints/unetmembranetest")

testGene = testGenerator("/home/m/m_kais13/unet/data/membrane/test")
results = model.predict_generator(testGene,30,verbose=1)
saveResult(resultpath,results)