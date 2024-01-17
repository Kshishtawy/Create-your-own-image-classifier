import argparse
import json
import model_lib
import data_lib

parser = argparse.ArgumentParser(description='Predict')

parser.add_argument('image_dir')
parser.add_argument('checkpoint')
parser.add_argument('--category_names',  default='cat_to_name.json')
parser.add_argument('--topk',  default= 1)
parser.add_argument('--gpu',  default= False)

args = parser.parse_args()

image_dir = args.image_dir
checkpoint = args.checkpoint
category_names = args.category_names
topk = int(args.topk)
gpu = args.gpu

if gpu:
    print('GPU: ON \n')
else:
    print('GPU: OFF \n')


# loading model
model = model_lib.load_model(checkpoint)

# predictt the class of an image
probs, classes = model_lib.predict(image_dir, model, topk, gpu)

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    names_of_classes = []
    for i in classes:
        names_of_classes.append(cat_to_name[i])
    print(probs)
    print(names_of_classes)