from django.shortcuts import render
from .models import image
from .form import imageform
import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy 
import torchvision
from torch.autograd import Variable
from PIL import Image



def index(request):
    #img=image.objects.all() 
    if request.method == "POST": 
        form=imageform(data=request.POST,files=request.FILES) 

        if form.is_valid(): 
            form.save()
            image1 = form.cleaned_data.get('image')
            classes = ['Buildings',
            'Forest',
            'Glaciers',
            'Mountains',
            'Sea',
            'Streets']


            model = torchvision.models.resnet18()
            #Parameters of newly constructed modules have requires_grad=True by default
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 6)

            model = model
            #checkpoint = torch.load('best_checkpoint.model', map_location=torch.device('cpu'))
            checkpoint = torch.load('99.5.model', map_location=torch.device('cpu'))

            model.load_state_dict(checkpoint)
            model.eval()
            

            transformer=transforms.Compose([
            transforms.Resize((150,150)),                         
            transforms.ToTensor(),  #from 0-255 color channel to 0-1, and transform numpy to tensors to use it in pytourch
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 0-1 to [-1,1] , formula (x-mean)/std x = old pixle value, (mean,std) = 0.5
                                
            ])

            def prediction(img_path,transformer):
                image2=Image.open(img_path)
                image_tensor=transformer(image2).float()
                image_tensor=image_tensor.unsqueeze_(0)
                #if torch.cuda.is_available():
                #image_tensor.cuda() 
                input=Variable(image_tensor)
                output=model(input)
                index=output.data.numpy().argmax()
                pred=classes[index]
                return pred
            

            label1 = prediction(image1,transformer)
            #obj = image.objects.create(image=image1,lable=label1)
            #obj.save()
            #img=image.objects.all() 
            img_obj = form.instance
            return render(request,"index.html",{"img_obj":img_obj ,"lable":label1 , "form": form})
    form=imageform()
    return render(request,"index.html",{"form":form}) 

    

# Create your views here.
