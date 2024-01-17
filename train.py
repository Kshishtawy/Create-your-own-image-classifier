import argparse
import model_lib
import data_lib

parser = argparse.ArgumentParser(description='Train')

parser.add_argument('data_directory')
parser.add_argument('--save_dir',  default='')
parser.add_argument('--arch',  default='vgg16')
parser.add_argument('--learning_rate',  default= 0.003)
parser.add_argument('--hidden_units',  default= 1024)
parser.add_argument('--epochs',  default= 1)
parser.add_argument('--gpu',  default= False)

args = parser.parse_args()

save_dir = args.save_dir
model_arch = args.arch
lr = float(args.learning_rate)
hidden_units = int(args.hidden_units)
epochs = int(args.epochs)

'''
For some reason, the parser considers the output of gpu as 
a string even though this isnt the case in predict.py

The code below makes sure that even if the user 
inputs "False," the system won't interpret it as a True
'''
if args.gpu.lower().capitalize() == 'True':
    gpu = True
else:
    gpu = False


# loading data
train_data, trainloader, valoader, testloader = data_lib.load_data(args.data_directory)

# building
model = model_lib.build(model_arch, hidden_units)
model.class_to_idx = train_data.class_to_idx

# training
model, criterion = model_lib.train(model, lr, epochs, gpu, trainloader, valoader)

# evaluating
model_lib.evaluate(model, testloader, criterion, gpu)

# saving
model_lib.save_model(model, model_arch, lr, hidden_units, epochs, save_dir) 
