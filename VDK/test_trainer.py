import unittest
import trainer
import torch
import numpy as np
import matplotlib.pyplot as plt 


def tensor_to_array(tensor):
    array = torch.squeeze(tensor).detach().numpy()
    return np.transpose (array ,(1,2,0))

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.my_trainer = trainer.Trainer()



    def test_train(self):
        _, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)

    
        for image, true_label in self.my_trainer.train_loader:
            ax1.imshow(tensor_to_array(trainer.normalize(image)))
            vector_scores = self.my_trainer.model(trainer.normalize(image))
            model_label = int(torch.argmax(vector_scores))

            if model_label != true_label:
                continue

            if true_label == trainer.target_class:
                continue

            row0 , col0 = np.random.randint(0,224-trainer.patch_size,2)
            print(image.shape)
            mask = torch.zeros(1,3,224,224)
            mask[0, : , row0:row0 + trainer.patch_size , col0: col0 + trainer.patch_size] = torch.ones(1,3,trainer.patch_size,trainer.patch_size)
            image[0, : , row0:row0 + trainer.patch_size , col0: col0 + trainer.patch_size] = self.my_trainer.patch

            image.requires_grad = True

            for i in range(10):
                vector_scores = self.my_trainer.model(trainer.normalize(image))

                loss = -torch.nn.functional.log_softmax(vector_scores, dim = 1)
                loss[0, trainer.target_class].backward()
                with torch.no_grad() : 
                    image-= torch.mul(mask, image.grad)
            self.my_trainer.patch = image.detach()[0, : , row0:row0 + trainer.patch_size , col0: col0 + trainer.patch_size]
            
            ax2.imshow(tensor_to_array(image))
            ax3.imshow(tensor_to_array(100*image.grad))
            ax4.imshow(tensor_to_array(self.my_trainer.patch))
            ax5.imshow(tensor_to_array(mask))
            plt.pause(0.01)

    def test_loader(self):
        for image, true_label in self.my_trainer.train_loader:
            plt.imshow(tensor_to_array(image))
            plt.pause(0.01)