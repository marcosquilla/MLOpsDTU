import sys
import argparse

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from data import mnist
from model import MyAwesomeModel

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.0001)
        parser.add_argument('--batch_size', default=64)
        parser.add_argument('--weight_decay', default=1e-5)
        parser.add_argument('--epochs', default=50)
        
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        #print(args)
        
        # TODO: Implement training loop here
        trainset, _ = mnist()
        num_classes = 10
        channels = 1
        height = trainset.data.shape[-1]
        width = trainset.data.shape[-2]
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        model = MyAwesomeModel(num_classes, channels, height, width, 
                                   [{'nFilters':20, 'kernel':6, 'stride':2, 'padding':2}],
                                   [32, 16])
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        fig, ax = plt.subplots()
        plt.ion()
        plt.show()
        losses = []
        model.train()
        
        for epoch in tqdm(range(args.epochs)):  # loop over the dataset multiple times
            running_loss = 0
            for inputs, labels in trainloader:
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                output = model(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            losses.append(running_loss/len(trainloader))
            fig.clf()
            ax.plot(losses)
            plt.pause(0.05)
        torch.save(model.state_dict(), 'model_state')
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="model_state")
        parser.add_argument('--batch_size', default=64)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        _, testset = mnist()
        num_classes = 10
        channels = 1
        height = testset.data.shape[-1]
        width = testset.data.shape[-2]
        
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)
        if args.load_model_from:
            model = MyAwesomeModel(num_classes, channels, height, width, 
                                       [{'nFilters':20, 'kernel':6, 'stride':2, 'padding':2}],
                                       [32, 16])
            model.load_state_dict(torch.load(args.load_model_from))
            
        model.eval()
        total=0
        for inputs, labels in testloader: 
            preds = np.argmax(model(inputs).detach().numpy(), axis=1)
            total += np.sum(labels.detach().numpy() == preds)
        print('Accuracy of the network on the {} test images: {:4.2f} %'.format(len(testloader), total/len(testloader)))
            
if __name__ == '__main__':
    TrainOREvaluate()