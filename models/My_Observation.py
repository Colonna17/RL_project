#OBSERVATION MODEL IS A TRANSPOSED CNN
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn


"""
Observation needs two different models. One is used to transform input images from the scene into latent encoding.
The second one does the opposite action, taking one latent encoding generated by agent's dreaming and decoding it 
back into an image. For this part a transposed CNN is used as stated in the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ObservationEnc(nn.Module):
    def __init__(self, stride_sz=2, hidden_sz=32, kernel_sz=5, img_size=(3, 64, 64)):
        super().__init__()
        self.hidden_sz = hidden_sz
        self.stride = stride_sz
        stride = 2
        k_size = 4
        self.size = img_size
        self.dim1 = img_size[0]
        self.dim2 = img_size[1]
        self.conv1 = nn.Conv2d(self.dim1, hidden_sz, k_size, stride)
        self.conv2 = nn.Conv2d(hidden_sz, hidden_sz * 2, k_size, stride)
        self.conv3 = nn.Conv2d(hidden_sz * 2, hidden_sz * 4, k_size, stride)
        self.conv4 = nn.Conv2d(hidden_sz * 4, hidden_sz * 8, k_size, stride)
        self.activation = nn.ReLU()

    def forward(self, input):
        batch_shape = input.shape[:-3]
        img_shape = input.shape[-3:]
        embed = self.activation(self.conv1(input.reshape(-1, *img_shape)))
        embed = self.activation(self.conv2(embed))
        embed = self.activation(self.conv3(embed))
        embed = self.activation(self.conv4(embed))
        embed = torch.reshape(embed, (*batch_shape, -1))
        return embed
    
    @property
    def embed_size(self):
        conv1_shape = conv_out_shape(self.size[1:], 0, 4, self.stride)
        conv2_shape = conv_out_shape(conv1_shape, 0, 4, self.stride)
        conv3_shape = conv_out_shape(conv2_shape, 0, 4, self.stride)
        conv4_shape = conv_out_shape(conv3_shape, 0, 4, self.stride)
        embed_size = 8 * self.hidden_sz * np.prod(conv4_shape).item()
        return embed_size


class ObservationDec(nn.Module):
    def __init__(self, hidden_sz=32, stride_sz=2, activation = nn.ReLU, embedding_size = 1024, img_size = (3,64,64),):
        super().__init__()
        self.hidden_sz = hidden_sz
        self.stride = stride_sz
        stride = 2
        padding = 0
        dim1 = img_size[0]
        dim2 = img_size [1]
        self.activation = nn.ReLU   
        self.size = img_size
        self.embedding_size = embedding_size
        conv1_shape = conv_out_shape((dim2 , dim2 ), padding, 6, stride)
        conv1_pad = output_padding_shape(
            (dim2 , dim2 ), conv1_shape, padding, 6, stride
        )
        conv2_shape = conv_out_shape(conv1_shape, padding, 6, stride)
        conv2_pad = output_padding_shape(
            conv1_shape, conv2_shape, padding, 6, stride
        )
        conv3_shape = conv_out_shape(conv2_shape, padding, 5, stride)
        conv3_pad = output_padding_shape(
            conv2_shape, conv3_shape, padding, 5, stride
        )
        conv4_shape = conv_out_shape(conv3_shape, padding, 5, stride)
        conv4_pad = output_padding_shape(
            conv3_shape, conv4_shape, padding, 5, stride
        )
        self.conv_shape = (hidden_sz*32, *conv4_shape)
        #self.linear = nn.Linear(embedding_size , hidden_sz*32* np.prod(conv4_shape).item())
        # ATTENTION !!!!! Qua faccio hard coding e metto il valore di embedding_size uguale 
        # a 230. Potrei doverlo cambiare usando un agente diverso
        self.linear = nn.Linear(230 , hidden_sz*32* np.prod(conv4_shape).item())

        self.decoder = nn.Sequential(nn.ConvTranspose2d(hidden_sz*32, hidden_sz*4, 5, stride,
                                                        output_padding=conv4_pad),
                                     self.activation(),
                                     nn.ConvTranspose2d(hidden_sz*4,hidden_sz*2, 5,stride,
                                                        output_padding=conv3_pad),
                                     self.activation(),
                                     nn.ConvTranspose2d(hidden_sz*2,hidden_sz, 6,stride,
                                                        output_padding=conv2_pad ),
                                     self.activation(),
                                     nn.ConvTranspose2d(hidden_sz, dim1, 6, stride,
                                                         output_padding=conv1_pad),
                                    )

    def forward(self, encoding):
        """
        INPUTS
        encoding: encoded observation from latent space, size(*batch_shape, embed_size)
        OUTPUTS
        observation: returns an image size tensor, size(*batch_shape, *self.shape)
        """
        x = encoding
        #print('ENCODING_SIZE:', encoding.size())
        batch_shape = x.shape[:-1] # 50 x 50
        embed_size = x.shape[-1] # 230
        squeezed_size = np.prod(batch_shape).item() #2500
        #print('EMBED_SIZE', embed_size)
        #print('SQUEEZED_SIZE:', squeezed_size)
        x = x.reshape(squeezed_size, embed_size) #2500x230
        #print('COSA ESCE DOPO IL RESHAPE? =', x.size())
        x = self.linear(x)
        x = torch.reshape(x, (squeezed_size, *self.conv_shape))
        x = self.decoder(x)
        mean = torch.reshape(x, (*batch_shape, *self.size))
        obs_dist = td.Independent(td.Normal(mean, 1), len(self.size))
        return obs_dist
    
    @property
    def embed_size(self):
        conv1_shape = conv_out_shape(self.size[1:], 0, 4, self.stride)
        conv2_shape = conv_out_shape(conv1_shape, 0, 4, self.stride)
        conv3_shape = conv_out_shape(conv2_shape, 0, 4, self.stride)
        conv4_shape = conv_out_shape(conv3_shape, 0, 4, self.stride)
        embed_size = 8 *  self.hidden_sz * np.prod(conv4_shape).item()
        return embed_size


def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2.0 * padding - (kernel_size - 1.0) - 1.0) / stride + 1.0)


def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1


def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)


def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(
        output_padding(h_in[i], conv_out[i], padding, kernel_size, stride)
        for i in range(len(h_in))
    )