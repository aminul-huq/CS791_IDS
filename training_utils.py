import torch,os
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import os, cv2
from os import path
import copy

best_acc = 0
best_loss = 10000000

def train(model,train_loader,criterion,optim,device,epochs):
    model.train()
    train_loss, total_correct, total = 0,0,0

    for i,(images,labels) in enumerate(tqdm(train_loader)):
                        
        images, labels = images.to(device), labels.to(device)
        optim.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()

        train_loss += loss.item() * images.size(0)
        _,predicted = torch.max(outputs.data,1)

        total_correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print("Epoch: [{}]  loss: [{:.2f}] Train Accuracy [{:.2f}] ".format(epochs+1,train_loss/len(train_loader),
                                                                               total_correct*100/total))

    return train_loss/len(train_loader), total_correct*100/total

def test(model,test_loader, criterion,optim,modelname,device,epochs):
    model.eval()
    global best_acc
    test_loss,total_correct, total = 0,0,0
    
    with torch.no_grad():
        for i,(images, labels) in enumerate(tqdm(test_loader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _,predicted = torch.max(outputs.data,1)
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = total_correct*100/total
        print("Epoch: [{}]  loss: [{:.2f}] Test Accuracy [{:.2f}] ".format(epochs+1,test_loss/len(test_loader),acc))  

       
        if acc > best_acc:
            print('Saving Best model...')
            state = {
                        'model':model.state_dict(),
                        'acc':acc,
                        'epoch':epochs,
                }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_point = './checkpoint/'
            if not os.path.isdir(save_point):
                os.mkdir(save_point)

            torch.save(state, save_point+modelname+'model.pth.tar')
            best_acc = acc
        
    return test_loss/len(test_loader),acc


def best_test(model,test_loader,criterion,optim,device,epochs):
    model.eval()
    test_loss,total_correct, total = 0,0,0
    y,y_pred,img = [],[],[]
    for i,(images,labels) in enumerate(tqdm(test_loader)):
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _,predicted = torch.max(outputs.data,1)
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)

            y.append(labels.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())
            
    acc = total_correct*100/total
    print("Epoch: [{}]  loss: [{:.2f}] Test Accuracy [{:.2f}] ".format(epochs+1,test_loss/len(test_loader),acc))    
    return test_loss/len(test_loader),acc,y,y_pred

def train_autoencoder(model,train_loader,criterion,optim,device,epochs):
    model.train()
    train_loss, total_correct, total = 0,0,0

    for i,(images,labels) in enumerate(tqdm(train_loader)):
                        
        images, labels = images.to(device), labels.to(device)
        optim.zero_grad()
        decoded_output, _, x, second_latent = model(images)
        
        loss = criterion(decoded_output, images) + criterion(x, second_latent)
        loss.backward()
        optim.step()

        train_loss += loss.item() * images.size(0)

    print("Epoch: [{}]  loss: [{:.2f}]".format(epochs+1,train_loss/len(train_loader)))

    return train_loss/len(train_loader)

def test_autoencoder(model,test_loader, criterion,optim,modelname,device,epochs):
    model.eval()
    global best_loss
    test_loss,total_correct, total = 0,0,0
    
    with torch.no_grad():
        for i,(images, labels) in enumerate(tqdm(test_loader)):
            images, labels = images.to(device), labels.to(device)
            decoded_output, _, x, second_latent = model(images)
        
            loss = criterion(decoded_output, images) + criterion(x, second_latent)

            test_loss += loss.item() * images.size(0)
        

        
        print("Epoch: [{}]  loss: [{:.2f}] ".format(epochs+1,test_loss/len(test_loader)))  

       
        if loss < best_loss:
            print('Saving Best model...')
            state = {
                        'model':model.state_dict(),
                }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_point = './checkpoint/'
            if not os.path.isdir(save_point):
                os.mkdir(save_point)

            torch.save(state, save_point+modelname+'model.pth.tar')
            best_loss = loss
        
    return test_loss/len(test_loader)


def best_test_autoencoder(model,test_loader,criterion,optim,device,epochs):
    model.eval()
    test_loss,total_correct, total = 0,0,0
    y,y_pred,img = [],[],[]
    for i,(images,labels) in enumerate(tqdm(test_loader)):
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            decoded_output, _, x, second_latent = model(images)
        
            loss = criterion(decoded_output, images) + criterion(x, second_latent)

            test_loss += loss.item() * images.size(0)
            
    print("Epoch: [{}]  loss: [{:.2f}] ".format(epochs+1,test_loss/len(test_loader)))    
    return test_loss/len(test_loader)


def train_both(model,train_loader,criterion1,criterion2,optim,device,epochs):
    model.train()
    train_loss, total_correct, total = 0,0,0

    for i,(images,labels) in enumerate(tqdm(train_loader)):
                        
        images, labels = images.to(device), labels.to(device)
        optim.zero_grad()
        reconstruction , outputs, first_latent, second_latent = model(images)
        
        loss1 = criterion1(outputs, labels)
        loss2 = criterion2(reconstruction, images)
        loss3 = criterion2(first_latent, second_latent)
        
        loss = 1*loss1 + 4*loss2 + 4*loss3        
        loss.backward()
        optim.step()

        train_loss += loss.item() * images.size(0)
        _,predicted = torch.max(outputs.data,1)

        total_correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print("Epoch: [{}]  loss: [{:.2f}] Train Accuracy [{:.2f}] ".format(epochs+1,train_loss/len(train_loader),
                                                                               total_correct*100/total))

    return train_loss/len(train_loader), total_correct*100/total


def test_both(model,test_loader, criterion1,criterion2,optim,modelname,device,epochs):
    model.eval()
    global best_acc
    test_loss,total_correct, total = 0,0,0
    
    with torch.no_grad():
        for i,(images, labels) in enumerate(tqdm(test_loader)):
            images, labels = images.to(device), labels.to(device)
            reconstruction, outputs, first_latent, second_latent = model(images)
            
            loss1 = criterion1(outputs, labels)
            loss2 = criterion2(reconstruction, images)
            loss3 = criterion2(first_latent, second_latent)
        
            loss = loss1 + loss2 + loss3
            test_loss += loss.item() * images.size(0)
            _,predicted = torch.max(outputs.data,1)
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = total_correct*100/total
        print("Epoch: [{}]  loss: [{:.2f}] Test Accuracy [{:.2f}] ".format(epochs+1,test_loss/len(test_loader),acc))  

       
        if acc > best_acc:
            print('Saving Best model...')
            state = {
                        'model':model.state_dict(),
                        'acc':acc,
                        'epoch':epochs,
                }
 
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_point = './checkpoint/'
            if not os.path.isdir(save_point):
                os.mkdir(save_point)

            torch.save(state, save_point+modelname+'model.pth.tar')
            best_acc = acc
        
    return test_loss/len(test_loader),acc


def best_test_both(model,test_loader, criterion1,criterion2,device):
    model.eval()
    global best_acc
    test_loss,total_correct, total = 0,0,0
    
    with torch.no_grad():
        for i,(images, labels) in enumerate(tqdm(test_loader)):
            images, labels = images.to(device), labels.to(device)
            reconstruction, outputs, _,_ = model(images)
            
            loss1 = criterion1(outputs, labels)
            loss2 = criterion2(reconstruction, images)
        
            loss = loss1 + loss2
            test_loss += loss.item() * images.size(0)
            _,predicted = torch.max(outputs.data,1)
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = total_correct*100/total
        print("Loss: [{:.2f}] Test Accuracy [{:.2f}] ".format(test_loss/len(test_loader),acc))  
    