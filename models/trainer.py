"""
File that drives training
"""
import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from models.dann import FeatureExtractor, DomainClassifier, LabelPredictor

class Trainer:

    def __init__(self):
        self.data = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Currently using {self.device}.")
        self.source_transform = transforms.Compose([
            transforms.Grayscale(),
            # KEY: Adopt cv2.Canny to obtain edges (simulate the 'known' target data)
            transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15, fill=(0,)),
            transforms.ToTensor(),
        ])
        self.target_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15, fill=(0,)),
            transforms.ToTensor(),
        ])

        self.source_dataset = ImageFolder('data/real_or_drawing/train_data', transform=self.source_transform)
        self.target_dataset = ImageFolder('data/real_or_drawing/test_data', transform=self.target_transform)
        
        self.source_dataloader = DataLoader(self.source_dataset, batch_size=32, shuffle=True)
        self.target_dataloader = DataLoader(self.target_dataset, batch_size=32, shuffle=True)
        self.test_dataloader = DataLoader(self.target_dataset, batch_size=128, shuffle=False)

        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.BCEWithLogitsLoss()

        self.feature_extractor = FeatureExtractor().to(self.device)
        self.label_predictor = LabelPredictor().to(self.device)
        self.domain_classifier = DomainClassifier().to(self.device)
        
        self.opt_F = optim.Adam(self.feature_extractor.parameters())
        self.opt_C = optim.Adam(self.label_predictor.parameters())
        self.opt_D = optim.Adam(self.domain_classifier.parameters())

    def train_epoch(self, source_dataloader, target_dataloader, lamb):
        '''
        lamb: control the balance of domain adaptation and classification.
        '''

        # D loss: Domain Classifier's loss
        # F loss: Feature Extrator & Label Predictor's loss
        running_D_loss, running_F_loss = 0.0, 0.0
        total_hit, total_num = 0.0, 0.0

        for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):

            source_data = source_data.to(self.device)
            source_label = source_label.to(self.device)
            target_data = target_data.to(self.device)
            
            # Mixed the source data and target data, or it'll mislead the running params of batch_norm.
            mixed_data = torch.cat([source_data, target_data], dim=0)
            domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).to(self.device)
            # set domain label of source data to be 1.
            domain_label[:source_data.shape[0]] = 1

            # Step 1 : train domain classifier
            feature = self.feature_extractor(mixed_data)
            # We don't need to train feature extractor in step 1.
            # Thus we detach the feature neurself.on to avoid backpropgation.
            domain_logits = self.domain_classifier(feature.detach())
            loss = self.domain_criterion(domain_logits, domain_label)
            running_D_loss+= loss.item()
            loss.backward()
            self.opt_D.step()

            # Step 2 : train feature extractor and label classifier
            class_logits = self.label_predictor(feature[:source_data.shape[0]])
            domain_logits = self.domain_classifier(feature)
            # loss = cross entropy of classification - lamb * domain binary cross entropy.
            #  The reason why using subtraction is similar to generator loss in disciminator of GAN
            loss = self.class_criterion(class_logits, source_label) - lamb * self.domain_criterion(domain_logits, domain_label)
            running_F_loss+= loss.item()
            loss.backward()
            self.opt_F.step()
            self.opt_C.step()

            self.opt_D.zero_grad()
            self.opt_F.zero_grad()
            self.opt_C.zero_grad()

            total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
            total_num += source_data.shape[0]
            print(i, end='\r')

        return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num
        
    def train(self, total_epochs=100):
        gamma = 10
        print(f"Total epoch: {total_epochs}.")
        for epoch in range(total_epochs):
            prog = epoch / total_epochs
            sch_lamb = 2 / (1 + np.exp(-gamma * prog)) - 1
            train_D_loss, train_F_loss, train_acc = self.train_epoch(
                self.source_dataloader, self.target_dataloader, lamb=sch_lamb)
            if epoch % 100 == 0 and epoch != 0:
                torch.save(self.feature_extractor.state_dict(), f'extractor_model_{epoch}.bin')
                torch.save(self.label_predictor.state_dict(), f'predictor_model_{epoch}.bin')
            print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss, train_F_loss, train_acc))

