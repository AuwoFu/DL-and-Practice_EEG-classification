import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset


import dataloader
from datetime import datetime
from tqdm import tqdm
# print #parameters of model
from torchinfo import summary

class EEGNet(nn.Module):
    def __init__(self,activate_func="ELU"):
        super(EEGNet, self).__init__()
        self.flatten = nn.Flatten()
        if activate_func=="LeakyReLU":
            self.activateion=nn.LeakyReLU()
        elif activate_func=="ReLU":
            self.activateion=nn.ReLU()
        else:
            self.activateion=nn.ELU()

        self.firstConv = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=(1,51),stride=(1,1),padding=(0,25),bias=False),
            nn.BatchNorm2d(16),
        )
        self.depthwiseConv=nn.Sequential(
            nn.Conv2d(16,32,kernel_size=(2,1),stride=(1,1),groups=16,bias=False),
            nn.BatchNorm2d(32),
            self.activateion,
            nn.AvgPool2d(kernel_size=(1,4),stride=(1,4),padding=0),
            nn.Dropout(p=0.25)
        )
        self.seperableConv=nn.Sequential(
            nn.Conv2d(32,32,kernel_size=(1,15),stride=(1,1),padding=(0,7),bias=False),
            nn.BatchNorm2d(32),
            self.activateion,
            nn.AvgPool2d(kernel_size=(1,8),stride=(1,8),padding=0),
            nn.Dropout(p=0.25)
        )
        self.flatten=nn.Flatten()
        self.classfy=nn.Sequential(
            nn.Linear(in_features=736, out_features=2,bias=True)
            #,nn.LogSoftmax() #for NLLLoss
        )


    def forward(self, x):
        x1=self.firstConv(x)
        x2=self.depthwiseConv(x1)
        x3=self.seperableConv(x2)
        x3 = self.flatten(x3)
        y=self.classfy(x3)
        return y

class DeepConvNet(nn.Module):
    def __init__(self,activate_func="ELU",C=2,T=750,N=2):
        super(DeepConvNet, self).__init__()
        
        if activate_func=="LeakyReLU":
            self.activateion=nn.LeakyReLU()
        elif activate_func=="ReLU":
            self.activateion=nn.ReLU()
        else:
            self.activateion=nn.ELU()

        self.Layer1 = nn.Sequential(
            nn.Conv2d(1,25,kernel_size=(1,5)),
            nn.Conv2d(25,25,kernel_size=(C,1)),
            nn.BatchNorm2d(25),
            self.activateion,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        )
        self.Layer2=nn.Sequential(
            nn.Conv2d(25,50,kernel_size=(1,5)),
            nn.BatchNorm2d(50),
            self.activateion,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        )
        self.Layer3=nn.Sequential(
            nn.Conv2d(50,100,kernel_size=(1,5)),
            nn.BatchNorm2d(100),
            self.activateion,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        )
        self.Layer4=nn.Sequential(
            nn.Conv2d(100,200,kernel_size=(1,5)),
            nn.BatchNorm2d(200),
            self.activateion,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        )
        self.flatten = nn.Flatten()

        self.Dense=nn.Sequential(
            nn.Linear(8600,N)
            #,nn.Softmax(dim=1) # ignore for Cross Entropy Loss
            #,nn.LogSoftmax() #for NLLLoss (need to ignore Softmax)          
        )

    def forward(self, x):
        x1=self.Layer1(x)
        x2=self.Layer2(x1)
        x3=self.Layer3(x2)
        x4=self.Layer4(x3)
        x4=self.flatten(x4)
        y=self.Dense(x4)
        return y


def training(model,training_loader,epoch=300):
    # hyper parameter
    loss_fn = nn.CrossEntropyLoss()
    #loss_fn=nn.NLLLoss()
    #loss_fn=nn.MSELoss()
    #loss_fn=nn.KLDivLoss()
    #loss_fn=nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    running_loss = 0.
    train_record,test_record=[],[]

    best_score=0
    best_model=None
    for epoch_index in (range(epoch)):
        running_loss=0
        model.train()
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            input_data, labels = data
            
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            input_data=input_data.to(device)
            labels=labels.to(device)
            outputs = model(input_data)

            # Compute the loss and its gradients
            # for MSELoss
            '''outputs=outputs.argmax(dim=1).to(dtype=torch.float)
            outputs.requires_grad=True
            labels=labels.to(dtype=torch.float)'''


            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            
        
        torch.save(model.state_dict(), PATH)


        # get accuracy
        acc=0
        model.eval()
        # train set
        for i, data in enumerate(training_loader):
            input_data, labels = data
            input_data=input_data.to(device)
            labels=labels.to(device)
            outputs = model(input_data)
            outputs=torch.argmax(outputs, dim=1)
            acc+=torch.sum(outputs == labels).item()
                
        train_acc=acc/1080*100
        train_record.append(train_acc)
        
        # test set
        acc=0
        for i, data in enumerate(test_loader):
            input_data, labels = data
            input_data=input_data.to(device)
            labels=labels.to(device)
            outputs = model(input_data)
            outputs=torch.argmax(outputs, dim=1)
            acc+=torch.sum(outputs == labels).item()
                
        test_acc=acc/1080*100
        test_record.append(test_acc)

        if epoch_index%10==9:
            print(f'epoch {epoch_index+1:5} loss: {running_loss:.6f} train:{train_acc:.4f} test:{test_acc:.4f}')
        if test_acc>best_score:
            best_model=model
            best_score=test_acc
    print(f'max test score= {best_score} at epoch{epoch_index+1}')
    return model,best_model,train_record,test_record,best_score





if __name__=="__main__":
    # set training device
    os.environ['CUDA_LAUNCH_BLOCKING']='1'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #print(f"Using {device} device")
    
    # load data
    train_data, train_label, test_data, test_label=dataloader.read_bci_data()

    # convert ndarray to pytorch tensor dataset
    train_data=torch.Tensor(train_data)
    train_label=torch.Tensor(train_label).to(dtype=torch.long)
    training_set=TensorDataset(train_data,train_label)
    
    test_data=torch.Tensor(test_data)
    test_label=torch.Tensor(test_label).to(dtype=torch.long)
    test_set=TensorDataset(test_data,test_label)

    # create dataloader
    batch_size=1080
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set,batch_size=batch_size, shuffle=False, num_workers=2)
    

    now = datetime.now()
    current_time = now.strftime("%d_%H-%M-%S")
    modelSaveDir=f'./{current_time}'
    if not os.path.exists(modelSaveDir):
        os.makedirs(modelSaveDir)

    # show structure
    eeg_model = EEGNet().to(device)
    #print(eeg_model)
    #summary(eeg_model, input_size=(1080, 1, 2, 750))
    deepConv_model=DeepConvNet().to(device)
    #print(deepConv_model)
    #summary(deepConv_model, input_size=(1080, 1, 2, 750))

    # record structure
    f=open(f'{modelSaveDir}/model_structure.txt','w')
    f.write(f'{eeg_model}\n{deepConv_model}\n')
    f.close()

    
    epoch_time=500
    X=[i+1 for i in range(epoch_time)]

    f = open("Experiment Result.txt", "a")
    f.write(f'\n{current_time}   epoch:{epoch_time} batch:{batch_size}\n')

    activate_func_list=["ELU","LeakyReLU","ReLU"]
    
    # EEGNet
    plt.clf()
    
    for actFunc in activate_func_list:
        PATH="./eeg_model"
        eeg_model = EEGNet(activate_func=actFunc).to(device)
        eeg_model,best_model,eeg_train,eeg_test,best_score=training(eeg_model,training_loader,epoch=epoch_time)
        
        torch.save(eeg_model,f'{modelSaveDir}/EEGNet_{actFunc}')
        torch.save(best_model,f'{modelSaveDir}/best_EEGNet_{actFunc}')
        
        plt.plot(X,eeg_train,'-',label=f"{actFunc}_train")
        plt.plot(X,eeg_test,'-',label=f"{actFunc}_test")
        f.write(f'EEG_{actFunc:15s} {eeg_test[-1]:.6f} best: {best_score}\n')
        print(f'EEG_{actFunc}: {eeg_test[-1]:.6f} best: {best_score}\n')
    f.close()

    plt.title("Sctivation Function Comparision_EEGNet")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.legend()
    plt.savefig(f'{modelSaveDir}/EEGNet')

    

    # DeepConvNet
    plt.clf()
    f = open("Experiment Result.txt", "a")
    for actFunc in activate_func_list:
        # DeepConvNet
        PATH="./deepConv_model"
        deepConv_model=DeepConvNet(activate_func=actFunc).to(device)
        deepConv_model,best_model,deepConv_train,deepConv_test,best_score=training(deepConv_model,training_loader,epoch=epoch_time)
        
        torch.save(deepConv_model,f'{modelSaveDir}/DeepConvNet_{actFunc}') 
        torch.save(best_model,f'{modelSaveDir}/best_DeepConvNet_{actFunc}')           
        
        plt.plot(X,deepConv_train,'-',label=f"{actFunc}_train")
        plt.plot(X,deepConv_test,'-',label=f"{actFunc}_test")
        f.write(f'deepConv_{actFunc:10s} {deepConv_test[-1]:.6f} best: {best_score}\n')
        print(f'deepConv_{actFunc}: {deepConv_test[-1]:.6f} best: {best_score}\n')
    f.close()
    plt.title("Sctivation Function Comparision_DeepConvNet")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.legend()
    plt.savefig(f'{modelSaveDir}/DeepConvNet')


    