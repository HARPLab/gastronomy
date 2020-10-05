import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AudioEmbeddingsNet(nn.Module):
    """
    Network for embedding audio data, forawrd model outputs an embedding
    Didn't use normal weight initialization like Kevin, just used pytorch default.
    To perform normal weight initialization: https://stackoverflow.com/a/55546528
    """
    def __init__(self, input_dim, output_dim, fc1_size=100, fc2_size=100):
        super(AudioEmbeddingsNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, output_dim) # TODO might want out dim to be the number of classes

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class ForceEmbeddingsNet(nn.Module):
    """
    Network for embedding force data, forawrd model outputs an embedding
    """
    def __init__(self, input_dim, output_dim, fc1_size=100, fc2_size=100):
        super(ForceEmbeddingsNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, output_dim) # TODO might want out dim to be the number of classes

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class ResNetEmbeddingsNet(nn.Module):
    """
    Embedding network for images based on ResNet architecture, image should be 224x224x3
    Assuming the input to forward is a sequence of the above mentioned image size
    Will pass each image through resnet and concatenate each sample invidually in the order given
    https://pytorch.org/docs/stable/torchvision/models.html
    """
    def __init__(self, out_size, fc1_size=1024, fc2_size=512, sequence_length=8, drop_prob=0.2):
        super(ResNetEmbeddingsNet, self).__init__()
        self.drop_prob = drop_prob
        self.sequence_length = sequence_length
        
        # get resnet module and delete the last fully connected layer
        resnet = models.resnet34(pretrained=True)
        modules = list(resnet.children())[:-1] # output is now size 512 for resnet34
        
        self.resnet = nn.Sequential(*modules)
        self.resnet_out_size = resnet.fc.in_features*sequence_length # 512xseq_length for resnet34
        self.fc1 = nn.Linear(self.resnet_out_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, out_size)

    def forward(self, x):
        """
        Inputs:
            x - (batch_size, sequence_length, D, H, W): Input sequence of images
        """
        assert x.size(1) == self.sequence_length
        output = []
        # cycle through the images in the sequence, embed and concatenate each sample
        for t in range(self.sequence_length):
            with torch.no_grad():
                #TODO decide on if you want to learn the weights for resnet too
                temp_x = self.resnet(x[:, t, :, :, :]).squeeze()
                output.append(temp_x)
        output = torch.cat(output, axis=-1)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output) # TODO decide on activation here

        return output

class CombineEmbeddingsNet(nn.Module):
    """
    Wrapper network for combining two embeddings to return a single embedding
    """
    def __init__(self, out_size, embed1_in_size, embed2_in_size,
                 fc1_size=512, fc2_size=256):
        super(CombineEmbeddingsNet, self).__init__()
        self.fc1 = nn.Linear((embed1_in_size + embed2_in_size), fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, out_size)

    def forward(self, embeddings_input1, embeddings_input2):
        # concatenate the two embeddings
        output = torch.cat((embeddings_input1, embeddings_input2), dim=-1)

        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output) # TODO decide on activation here

        return output

class SiameseNet(nn.Module):
    """
    Wrapper network to process pairs of inputs for a given embedding netork 
    Inputs:
        embedding_net(nn.Module): the base network to run each of the inputs through
        multi_args(bool): set flag to true if the embeddings net forward takes multiple
            arguments and the inputs you will use for this forward are all tuples
    Ref: https://github.com/adambielski/siamese-triplet/blob/master/networks.py
    """
    def __init__(self, embedding_net, multi_args=True):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.multi_args = multi_args

    def forward(self, x1, x2):
        if self.multi_args:
            output1 = self.embedding_net(*x1)
            output2 = self.embedding_net(*x2)
        else:
            output1 = self.embedding_net(x1)
            output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)

class TripletNet(nn.Module):
    """
    Wrapper network to process triplets of inputs for a given embedding netork 
    Inputs: 
        embedding_net(nn.Module): the base network to run each of the inputs through
        multi_args(bool): set flag to true if the embeddings net forward takes multiple
            arguments and the inputs you will use for this forward are all tuples
    Ref: https://github.com/adambielski/siamese-triplet/blob/master/networks.py
    """
    def __init__(self, embedding_net, multi_args=True):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net
        self.multi_args = multi_args

    def forward(self, x1, x2, x3):
        if self.multi_args:
            output1 = self.embedding_net(*x1)
            output2 = self.embedding_net(*x2)
            output3 = self.embedding_net(*x3)
        else:
            output1 = self.embedding_net(x1)
            output2 = self.embedding_net(x2)
            output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


class ClassificationNet(nn.Module):
    """
    Wrapper network to get class predictions for a given embedding
    Ref: https://github.com/adambielski/siamese-triplet/blob/master/networks.py
    """
    def __init__(self, n_classes, input_dim):
        super(ClassificationNet, self).__init__()
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(input_dim, n_classes)

    def forward(self, embedding_input):
        output = self.nonlinear(embedding_input)
        #TODO might wnt to change this activation, nn.CrossentropyLoss already uses LogSoftmax inside it
        scores = F.log_softmax(self.fc1(output), dim=-1) 
        return scores

class FoodEmbeddingsNet(nn.Module):
    """
    Network for making the embeddings and predictions for the audio, image, and force data.

    Inputs:
        n_classes(int): the total number of class labels
        audio_in_size(int): the input size of the audio data
        force_in_size(int): the input size of the force data
        audio_embed_size(int): the size to use for embedding the audio data
        audio_fc1_size(int): the size of the first linear/dense layer of the audio
            embeddings network
        audio_fc2_size(int): the size of the second linear/dense layer of the audio
            embeddings network
        image_embed_size(int): the size to use for embedding the image data
        image_fc1_size(int): the size of the first linear/dense layer of the image
            embeddings network
        image_fc2_size(int): the size of the second linear/dense layer of the image
            embeddings network
        force_embed_size(int): the size to use for embedding the force data
        force_fc1_size(int): the size of the first linear/dense layer of the force
            embeddings network
        force_fc2_size(int): the size of the second linear/dense layer of the force
            embeddings network
        audio_force_embed_size(int): the size to use for embedding the force and
            audio embeddings together
        audio_force_fc1_size(int): the size of the first linear/dense layer of the
            combined force & audio embeddings network
        audio_force_fc2_size(int): the size of the second linear/dense layer of the
            combined force & audio embeddings network
        final_fc1_size(int): the size of the first linear/dense layer of the
            combined force, audio & image embeddings network
        final_fc2_size(int): the size of the second linear/dense layer of the
            combined force, audio & image embeddings network
    """
    def __init__(self, n_classes, audio_in_size, force_in_size,
                 final_embed_size=512, 
                 audio_embed_size=128, audio_fc1_size=256, audio_fc2_size=128,
                 image_embed_size=1024, image_fc1_size=2048, image_fc2_size=1024,
                 force_embed_size=128, force_fc1_size=256, force_fc2_size=128,
                 audio_force_embed_size=128,
                 audio_force_fc1_size=256, audio_force_fc2_size=128,
                 final_fc1_size=1024, final_fc2_size=512):
        super(FoodEmbeddingsNet, self).__init__()
        self.audio_net = AudioEmbeddingsNet(input_dim=audio_in_size,
                                            output_dim=audio_embed_size,
                                            fc1_size=audio_fc1_size,
                                            fc2_size=audio_fc2_size)
        self.image_net = ResNetEmbeddingsNet(out_size=image_embed_size,
                                             fc1_size=image_fc1_size,
                                             fc2_size=image_fc2_size)
        self.force_net = ForceEmbeddingsNet(input_dim=force_in_size,
                                            output_dim=force_embed_size,
                                            fc1_size=force_fc1_size,
                                            fc2_size=force_fc2_size)

        self.audio_force_net = CombineEmbeddingsNet(out_size=audio_force_embed_size,
                                                    embed1_in_size=audio_embed_size,
                                                    embed2_in_size=force_embed_size,
                                                    fc1_size=audio_force_fc1_size,
                                                    fc2_size=audio_force_fc2_size)
        self.final_embeddings_net = CombineEmbeddingsNet(out_size=final_embed_size,
                                                        embed1_in_size=image_embed_size,
                                                        embed2_in_size=audio_force_embed_size,
                                                        fc1_size=final_fc1_size,
                                                        fc2_size=final_fc2_size)

        self.audio_classifier_net = ClassificationNet(n_classes=n_classes,
                                                      input_dim=audio_embed_size)
        self.image_classifier_net = ClassificationNet(n_classes=n_classes,
                                                      input_dim=image_embed_size)
        self.force_classifier_net = ClassificationNet(n_classes=n_classes,
                                                      input_dim=force_embed_size)
        self.final_classifier_net = ClassificationNet(n_classes=n_classes,
                                                      input_dim=final_embed_size)

    def forward(self, audio_input, image_input, force_input, return_embedding=True):
        audio_out = self.audio_net(audio_input)
        image_out = self.image_net(image_input)
        force_out = self.force_net(force_input)

        audio_force_out = self.audio_force_net(audio_out, force_out)
        final_out = self.final_embeddings_net(image_out, audio_force_out)

        audio_pred = self.audio_classifier_net(audio_out)
        image_pred = self.image_classifier_net(image_out)
        force_pred = self.force_classifier_net(force_out)
        final_pred = self.final_classifier_net(final_out)

        if return_embedding:
            return audio_pred, image_pred, force_pred, final_pred, final_out
        else:
            return audio_pred, image_pred, force_pred, final_pred
