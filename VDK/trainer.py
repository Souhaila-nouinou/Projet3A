
import torch
import torchvision
import matplotlib.pyplot as plt 
import numpy as np 
import math
from PIL import Image 

path_best_model = "VDK_model.pth"
n_classes = 5 
path_train_dataset = "C:\\Users\\dell\\projet3A\\new-fruit\\train"
path_test_dataset = "C:\\Users\\dell\\Desktop\\proj 3A\\new-fruit\\val"
relative_size = 0.05
patch_size = int(math.sqrt(224*224*relative_size))
target_class = 1 

normalize = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


class Trainer:
    def __init__(self):
        self.model = torchvision.models.alexnet(pretrained=True)
        self.model.classifier[6] = torch.nn.Linear(self.model.classifier[6].in_features, n_classes)
        self.model.load_state_dict(torch.load(path_best_model))

        train_dataset =  torchvision.datasets.ImageFolder(
            path_train_dataset,
            torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
            ])
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
        )


        test_dataset =  torchvision.datasets.ImageFolder(
        path_test_dataset,
        torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
        ]))

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=True,
        )

        self.patch = torch.rand(1, 3, patch_size, patch_size)


    def train_patch(self):
        c=0

        for image, true_label in self.train_loader:
            vector_scores = self.model(normalize(image))
            model_label = int(torch.argmax(vector_scores))

            if model_label != true_label:
                continue

            if true_label == target_class:
                continue

            row0 , col0 = np.random.randint(0,224-patch_size,2)
            mask = torch.zeros(1,3,224,224)
            mask[0, : , row0:row0 + patch_size , col0: col0 + patch_size] = torch.ones(1,3,patch_size,patch_size)
            # image[0, : , row0:row0 + patch_size , col0: col0 + patch_size] = self.patch
            patch_container = torch.zeros(1,3,224,224)
            patch_container[0, : , row0:row0 + patch_size , col0: col0 + patch_size] = self.patch


            for i in range(10):
                attacked = torch.mul(image, 1-mask) + torch.mul(patch_container, mask)
                attacked.requires_grad = True

                vector_scores = self.model(attacked)

                vector_proba = torch.nn.functional.softmax(vector_scores, dim=1)
                target_proba = float(vector_proba[0][target_class])
                print("image %d - iteration %d - target proba %f" % (c, i, target_proba))
                

                loss = -torch.nn.functional.log_softmax(vector_scores, dim = 1)
                loss[0, target_class].backward()

                patch_container-= attacked.grad


            self.patch = patch_container[0, : , row0:row0 + patch_size , col0: col0 + patch_size]

            if c%3:
                torchvision.utils.save_image(attacked, "img\\img_%d.jpeg" % c)

            if c>10:
                break
            c+=1



    def test_patch(self):
        successes ,total=0,0
        for image, true_label in self.test_loader:
            vector_scores = self.model(normalize(image))
            model_label = int(torch.argmax(vector_scores))

            if model_label != true_label:
                continue

            if true_label == target_class:
                continue
            
            row0 , col0 = np.random.randint(0,224-patch_size,2)
            image[0, : , row0:row0 + patch_size , col0: col0 + patch_size] = self.patch

            vector_scores = self.model(normalize(image))
            model_label = int(torch.argmax(vector_scores))

            if model_label == target_class:
                successes +=1

            total+=1 
            print("image %d success rate %f" % (total , (successes / total)*100 )  )
            if total>10:
                break 
        self.success_rate = (successes / total)*100




        

    
    


