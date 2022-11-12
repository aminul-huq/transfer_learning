import torch,os
from tqdm import tqdm

best_acc = 0


def train(model,train_loader,criterion,optim,device,epochs):
    model.train()
    train_loss, total_correct, total = 0,0,0

    for i,(images,labels) in enumerate(tqdm(train_loader)):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()

        train_loss += loss.item()
        _,predicted = torch.max(outputs.data,1)

        total_correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print("Epoch: [{}]  loss: [{:.2f}] Train Accuracy [{:.2f}] ".format(epochs+1,train_loss/len(train_loader),
                                                                               total_correct*100/total))

    return train_loss/len(train_loader), total_correct*100/total

def test(model,test_loader, criterion,optim,filename,modelname,device,epochs):
    model.eval()
    #global best_acc
    test_loss,total_correct, total = 0,0,0
    
    with torch.inference_mode():
        for i,(images, labels) in enumerate(tqdm(test_loader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _,predicted = torch.max(outputs.data,1)
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = total_correct*100/total
        print("Epoch: [{}]  loss: [{:.2f}] Test Accuracy [{:.2f}] ".format(epochs+1,test_loss/len(test_loader),acc))  

        if filename!=None:
            f = open(filename+".txt","a+")
            f.write('Epoch: [{}]  loss: [{:.2f}] Test Accuracy [{:.2f}]\n'.format(epochs+1,test_loss/len(test_loader),acc))
            f.close()


            if acc > best_acc:
                print('Saving Best model...')
                state = {
                            'model':model.state_dict(),
                            'acc':acc,
                            'epoch':epochs,
                    }
         
        if modelname!=None:       
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
    y,y_pred = [],[]
    for i,(images,labels) in enumerate(tqdm(test_loader)):
        images, labels = images.to(device), labels.to(device)
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