import numpy as np
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
import torchvision.models as models
import sys
import pickle
import os
import ipdb
import torch.nn.functional as F
st = ipdb.set_trace

device = torch.device("cuda")

params = {
    'clevr': {
        'num_classes': 41,
        'lr': 0.001,
        'datafile' : '/home/shamitl/datasets/clevr_veggie_baseline/clever_fewshot.p',
        'epochs': 10000,
        'val_every': 1,
        'log_every': 1,
        'checkpoint_every':250
    },
    'carla': {
        'num_classes': 20
    }
}

def process_data(f, num_classes):
    train_classes = f['train_class']
    train_rgbs = f['train_rgbs']
    test_classes = f['test_class']
    test_rgbs = f['test_rgbs']
    di = {}
    cnt = 0

    for train_class in train_classes:
        if train_class not in di:
            di[train_class] = cnt
            cnt += 1

    train_labels_list = []
    test_labels_list = []
    # for train_class in train_classes:
    #     label = di[train_class]
    #     onehot = np.zeros(num_classes)
    #     onehot[label] = 1
    #     train_labels_list.append(onehot)
    
    # for test_class in test_classes:
    #     label = di[test_class]
    #     onehot = np.zeros(num_classes)
    #     onehot[label] = 1
    #     test_labels_list.append(onehot)

    for train_class in train_classes:
        label = di[train_class]
        train_labels_list.append(label)
    
    for test_class in test_classes:
        label = di[test_class]
        test_labels_list.append(label)
        

    train_labels = torch.tensor(np.stack(train_labels_list)).to(device)
    test_labels = torch.tensor(np.stack(test_labels_list)).to(device)

    train_rgbs = F.interpolate(torch.tensor(np.stack(train_rgbs)).to(device), size=224)
    test_rgbs = F.interpolate(torch.tensor(np.stack(test_rgbs)).to(device), size=224)
    # st()
    return train_rgbs.float(), train_labels.long(), test_rgbs.float(), test_labels.long()






def main(exp_name, dataset_name):
    

    param = params[dataset_name]
    num_classes = param['num_classes']
    lr = param['lr']
    datafile = param['datafile']
    epochs = param['epochs']

    data = pickle.load(open(datafile, 'rb'))

    train_rgbs, train_labels, test_rgbs, test_labels = process_data(data, num_classes)

    
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, param['num_classes'])
    
    writer = SummaryWriter(os.path.join("baseline", "dataset_{}".format(dataset_name), exp_name))
    
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if not os.path.exists("baseline_checkpoints"):
        os.mkdir("baseline_checkpoints")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forward pass
        output = model(train_rgbs)
        # Calculate the loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, train_labels)
        loss.backward()
        # Optimizer takes one step
        optimizer.step()

        # Log info
        if epoch % param['log_every'] == 0:
            # todo: add your visualization code
            writer.add_scalar('Train Loss', loss.item(), epoch)

        
        # Validation iteration
        if epoch % param['val_every'] == 0:
            model.eval()
            with torch.no_grad():
                test_output = model(test_rgbs)
                test_preds = torch.argmax(test_output, dim=1)
                corr_preds = test_preds == test_labels
                test_acc = 1.*torch.sum(corr_preds)/len(test_labels)

                train_preds = torch.argmax(output, dim=1)
                corr_preds = train_preds == train_labels
                train_acc = torch.sum(corr_preds)/len(train_labels)
                # st()
            writer.add_scalar('Test accuracy', test_acc.item(), epoch)
            writer.add_scalar('Train accuracy', train_acc.item(), epoch)
            # print("Test accuracy is ",test_acc)
            # print("Train accuracy is: ", train_acc)
            print('Train Epoch: {}, Train Loss {} Train accuracy {} Test accuracy {}'.format(str(epoch), str(loss.item()), str(train_acc.item()), str(test_acc.item())))
            model.train()
        
        if epoch % param['checkpoint_every'] == 0:
            save_model(epoch, model, optimizer, epoch, exp_name, dataset_name)

def save_model(cnt, model, optimizer, epoch, exp_name, dataset_name):
    filename = "dataset_" + str(dataset_name) + "_expname_" + exp_name + '_model_%06d{}'.format('.pth') % cnt
    torch.save({"cnt": cnt, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch}, os.path.join("baseline_checkpoints", filename))

    
    
    



if __name__ == '__main__':
    exp_name = sys.argv[1]
    dataset_name = sys.argv[2]
    main(exp_name, dataset_name)
