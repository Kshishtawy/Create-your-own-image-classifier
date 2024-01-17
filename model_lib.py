import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from PIL import Image
import numpy as np
import seaborn as sns

def build(model_arch, hidden_units):
    print("Building the model...\n")
    if model_arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        input_units = 25088
    elif model_arch == 'densenet121':
        model = models.densenet121(pretrained = True)
        input_units = 1024
    elif model_arch == 'alexnet':
        model = models.alexnet(pretrained = True)
        input_units = 9216
    else:
        print("Model name is not definied. Using vgg16 architecture instead")
        model = models.vgg16(pretrained = True)
        input_units = 25088
              
    # Freezing parameters
    for param in model.parameters():
        param.requires_grad = False

    # Creating our classifier and Changing the model's classifer for our own
    model.classifier = nn.Sequential(nn.Linear(input_units, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2), # the dropout rate
                                     nn.Linear(hidden_units, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.2), # the second dropout rate
                                     nn.Linear(256, 102),
                                     nn.LogSoftmax(dim=1))
    print("The model has been built!\n")

    return model

def train(model, lr, epochs, gpu, trainloader, valoader):
    print("Training the model...\n")
    
    if gpu:
        print("GPU: ON \n")
        device = "cuda"
    else:
        print("GPU: OFF \n")
        device = "cpu"
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.003)

    model.to(device)
    
    steps = 0
    train_loss = 0
    print_every = 10

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1

            # move input and label to device (probably gpu)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0

                model.eval()

                with torch.no_grad():
                    for inputs, labels in valoader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        val_loss += batch_loss.item()

                        # accuracy
                        ps = torch.exp(logps) 
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {train_loss/print_every:.3f}.. "
                      f"Validation loss: {val_loss/len(valoader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valoader) * 100:.3f}")

                train_loss = 0
                model.train()
                
    print("The model finished training \n")

    return model, criterion

def evaluate(model, testloader, criterion, gpu):
    print("Evaluating the model on the test data...\n")
    
    if gpu:
        device = "cuda"
    else:
        device = "cpu"
        
    model.to(device)
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)

            # accuracy
            ps = torch.exp(logps) 
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"test accuracy: {accuracy/len(testloader) * 100:.3f}")
    
    
def save_model(model, model_arch, lr, hidden_units, epochs, save_dir):
    print("Saveing the mode...\n")
    
    checkpoint = {'model_state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'model_arch': model_arch,
                  'learning_rate': lr,
                  'hidden_units': hidden_units, 
                  'epochs': epochs,    
             }
    
    checkpoint_path = save_dir + "/" + 'checkpoint.pth'
    torch.save(checkpoint, checkpoint_path)
    
    print("Model saved to: ", "/" + checkpoint_path)

def load_model(filepath):
    print("Loading the model...\n")
    
    checkpoint = torch.load(filepath)
    
    arch = checkpoint['model_arch']
    hidden_units = checkpoint['hidden_units']
    model_state_dict = checkpoint['model_state_dict']
    class_to_idx = checkpoint['class_to_idx']
    
    model = build(arch, hidden_units)
    model.load_state_dict(model_state_dict)
    model.class_to_idx = class_to_idx
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    image = Image.open(image)
    
    transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                          std =  [0.229, 0.224, 0.225])
                                    ])
    edited_image = transform(image)
    
    return edited_image

def predict(image_path, model, topk, gpu):
    model.eval()
    
    if gpu:
        device = "cuda"
    else:
        device = "cpu"
    
    image = process_image(image_path).unsqueeze(0).float()
    
    image = image.to(device)
    model = model.to(device)
    
    with torch.no_grad():
        output = model.forward(image)

        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk)
        
        class_to_idx_inverted = {v : k for k, v in model.class_to_idx.items()}
        
        classes = []
        
        for top_class in top_class.tolist()[0]: 
            classes.append(class_to_idx_inverted[top_class])
        
        return top_p.tolist()[0], classes
          
          
            
              