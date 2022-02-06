from model import *
from data import *
import argparse

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#testcomment 

parser = argparse.ArgumentParser(description='set Hyperparameters for training')
parser.add_argument('--e' , '--epochs', type=int, metavar='', nargs='?', default=5, const=5, help='Number of Epochs')
parser.add_argument('--s' , '--steps', type=int, metavar='', nargs='?', default=300, const=300, help='Number of Steps per Epoch')
parser.add_argument('--bs' , '--batchsize', type=int, metavar='',nargs='?', default=2, const=2, help='Batch Size')
parser.add_argument('--lf' , '--lossfunction', metavar='',nargs='?', default='binary_crossentropy', const='binary_crossentropy', help='Loss Function for the Model')
parser.add_argument('--opt' , '--optimizer', metavar='',nargs='?', default=Adam(lr = 1e-4), const=Adam(lr = 1e-4), help='Optimizer Function for the model')
args = parser.parse_args()

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')


myGene = trainGenerator(args.batchsize,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

model = unet(args.lossfunction)
model_checkpoint = ModelCheckpoint('$WORK/checkpoints/unet_membranetest.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=args.steps,epochs=args.epochs,callbacks=[model_checkpoint])
#model.fit_generator(myGene,steps_per_epoch=args.steps,epochs=args.epochs)

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene,30,verbose=1)
#saveResult("data/membrane/test",results)