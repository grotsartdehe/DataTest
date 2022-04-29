#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.animation as animation
import matplotlib.image as mgimg
import time
import sys


# In[2]:


#classes utilitaires
class LinearEmbedding(nn.Module):
    def __init__(self, inp_size,d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * np.sqrt(self.d_model)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        
        pe = torch.zeros(max_seq_len, d_model)
        position = np.arange(0,max_seq_len)
        indexes = np.arange(0,d_model//2)
        angles = np.power(10000, 2*indexes/d_model)
        X,Y = np.meshgrid(angles,position)
        freq = Y/X
        pe[:, 0::2] = torch.sin(torch.from_numpy(freq))
        pe[:, 1::2] = torch.cos(torch.from_numpy(freq))       
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)#x.size(1)
        #x = x + self.pe.cuda()#Variable(self.pe[:,:seq_len],requires_grad=False).cuda()
        x += Variable(self.pe,requires_grad=False).cuda()
        return x
        
class LayerNorm(nn.Module):#couche Add and norm
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class FC(nn.Module):#fully connected layer du MLP
    def __init__(self, in_size, out_size, dropout_r, use_relu):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU(inplace=True) #inplace permet de gagner de la memoire en "overwrite" l input
        #self.gelu = nn.GELU()
        #on pourrait faire un if qui definit self.act_fct
        #comme ca dans le forward, il n y a pas de if
        """
        if use_relu:
            self.relu = nn.ReLU(inplace=True) #inplace permet de gagner de la memoire en "overwrite" l input 
        else:
            self.act_fct = nn.GELU()
        """
        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x) #implementation de la fully connected layer 

        if self.use_relu:
            #x = self.relu(x)
            x = self.relu(x)
            
        #si use_relu = False, on veut une fonction d'activation lineaire, donc on n'applique aucune fonction d'activation
        if self.dropout_r > 0:
            x = self.dropout(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r, use_relu):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, args):
        super(MHAtt, self).__init__()
        self.args = args
        self.linear_v = nn.Linear(args.HIDDEN_SIZE, args.HIDDEN_SIZE)
        self.linear_k = nn.Linear(args.HIDDEN_SIZE, args.HIDDEN_SIZE)
        self.linear_q = nn.Linear(args.HIDDEN_SIZE, args.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(args.HIDDEN_SIZE, args.HIDDEN_SIZE)

        self.dropout = nn.Dropout(args.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        #l embedding size doit etre divisible par le nombre de head 
        d_k = args.HIDDEN_SIZE // args.MULTI_HEAD #depth = split head
        
        #Les abstractions V,K,Q sont reshape 
        #Au depart, elles sont de taille Batch x seq length x emb size
        #elles sont transformees en Batch x N_head x seq length x emb size / N_head
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.args.MULTI_HEAD,
            d_k
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.args.MULTI_HEAD,
            d_k
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.args.MULTI_HEAD,
            d_k
        ).transpose(1, 2)
        
        atted = self.att(v, k, q, mask)#calcul l attention de la sequence
        
        #on reshape a nouveau pour obtenir une taille batch x seq length x emb size 
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.args.HIDDEN_SIZE
        )
        
        atted = self.linear_merge(atted)#quelle utilite?

        return atted

    def att(self, value, key, query, mask):
        d_k =query.size(-1)
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)
        #aux endroits ou mask = true, la fonction masked_fill remplit le score par -1e9
        #apres le softmax, cela vaudra 0
        
        if mask is not None:
            #scores = scores.masked_fill(mask, -1e9)
            #scores = scores.masked_fill(mask == 0, -1e9)
            scores = scores.masked_fill(mask == False, -1e9)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        return torch.matmul(att_map, value)

# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, args):
        super(FFN, self).__init__()
        self.mlp = MLP(
            in_size=args.HIDDEN_SIZE,
            mid_size=args.FF_SIZE,
            out_size=args.HIDDEN_SIZE,
            dropout_r=args.DROPOUT_R,
            use_relu=True
        )
    def forward(self, x):
        return self.mlp(x)

#block encodeur du Transformer
#4 sous-couches:MHA-norm-FFN-norm
class Block_encoder(nn.Module):
    def __init__(self, args):
        super(Block_encoder, self).__init__()

        self.mhatt = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.DROPOUT_R)
        self.norm1 = LayerNorm(args.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(args.DROPOUT_R)
        self.norm2 = LayerNorm(args.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        y = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))
        
        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))
        
        return y

class Block_decoder(nn.Module):
    def __init__(self, args):
        super(Block_decoder, self).__init__()

        self.mhatt1 = MHAtt(args)
        self.mhatt2 = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.DROPOUT_R)
        self.norm1 = LayerNorm(args.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(args.DROPOUT_R)
        self.norm2 = LayerNorm(args.HIDDEN_SIZE)
        
        self.dropout3 = nn.Dropout(args.DROPOUT_R)
        self.norm3 = LayerNorm(args.HIDDEN_SIZE)

    def forward(self, tgt, tgt_mask, x, x_mask):
        y = self.norm1(tgt + self.dropout1(
            self.mhatt1(tgt, tgt, tgt, tgt_mask)
        ))
        
        y = self.norm2(y + self.dropout2(
            self.mhatt2(x, x, y, x_mask)
        ))
        """
        y = self.norm2(y + self.dropout2(
            self.mhatt2(x, x, tgt, x_mask)
        ))
        """
        y = self.norm3(y + self.dropout3(
            self.ffn(y)
        ))
        
        return y
        
# -------------------------
# ---- Main Net Model ----
# -------------------------
   
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.embed = LinearEmbedding(1,args.HIDDEN_SIZE)
        self.embed_dec = LinearEmbedding(1,args.HIDDEN_SIZE)
        self.pos = PositionalEncoder(args.HIDDEN_SIZE,args.SEQ_LENGTH)
        self.enc = nn.ModuleList([Block_encoder(args) for _ in range(args.LAYER)])#plusieurs encodeurs dans le cas multimodal
        self.dec = nn.ModuleList([Block_decoder(args) for _ in range(args.LAYER)])
    def forward(self, x, x_mask, tgt, tgt_mask):
        # Transformer encoder
        #on boucle sur les N couches d'encodage
        #on ne garde que la sortie de la derniere couche
        self.pos_dec = PositionalEncoder(args.HIDDEN_SIZE,tgt.shape[1])
        x = self.embed(x)
        x = self.pos(x)#normalement, c'est ok: pas besoin de boucle pour ajouter a chaque patch
        #print("x shape", x.shape)
        #print("y shape", tgt.shape)
        tgt = torch.unsqueeze(tgt,2)
        tgt = self.embed_dec(tgt)
        tgt = self.pos_dec(tgt)
        
        self.final_layer = nn.Linear(args.HIDDEN_SIZE ,tgt.shape[1]).cuda()
        for enc in self.enc:
            y = enc(x, x_mask)
        e_outputs = y    
        
        for dec in self.dec:
            y = dec(tgt, tgt_mask, e_outputs, x_mask)
        
        #y = y.reshape(y.shape[0],y.shape[1]*y.shape[2])
        #print("out shape", y.shape)
        out = self.final_layer(y[:,0,:]).float()#besoin du float?
        return out


# In[3]:


class Mydataset():
    def __init__(self,X_new, Y_new):
        self.x = X_new
        self.y = Y_new
 
    def __len__(self):
        return self.x.shape[0]
 
    def __getitem__(self, idx):
        #return self.x[idx], self.y[idx,:]
        return self.x[idx,:,:], self.y[idx,:]
    
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data,device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self,dl,device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
        self.end_value = b
    def __len__(self):
        return len(self.dl)
    def __end__(self):
        return self.end_value


# In[4]:


#creation des donnees 
def events(L,Tend):#genere des variables suivant une loi de exponentielle
    timestamp = [0]
    U = np.random.uniform(0,1)
    t1 = -np.log(U)/L
    while t1 <= Tend:
        timestamp.append(t1)
        U = np.random.uniform(0,1)
        t1 += -np.log(U)/L
    return timestamp

def vectorSimulation(L,diff,Tend,t):
    timestamp = events(L,Tend)        
    y = np.zeros(len(t))
    i = 0
    while i < len(timestamp):
        if i+2 < len(timestamp):
            a = np.random.uniform(-diff,diff)
            t1 = timestamp[i]
            t2 = timestamp[i+1]
            y[(t>=t1) & (t<=t2)] = a
        i+=2
    #noise = np.random.normal(0,diff,len(t))
    #y += noise
    return y

def signal(A):
    Tend = 500
    step = 0.2
    amp= A
    dA = 5
    T = 4.0
    f = 1/T
    dS = 5
    df = 0.5
    L = 2/30
    
    t = np.arange(0,Tend,step)
    
    amp += vectorSimulation(L,dA,Tend,t)
    s = vectorSimulation(L,dS,Tend,t)
    f += vectorSimulation(L,df,Tend,t)
    
    y = amp*np.sin(2*np.pi*f*(t%(1/f))) + s
    return y

def association(x,dim):
    dimension = np.abs(dim)
    data = np.zeros((len(x),dimension))
    for i in range(len(x)):
        for j in range(dimension):
            if dim > 0:#on cherche des valeurs jusque t + dim
                if i < len(x)-dimension:
                    data[i,j] = x[i+1+j]
                else:
                    data[i,:] = x[i]
            else: #on prend les valeurs de t - dim a t 
                if i < len(x)-dimension:
                    data[i,dimension-1-j] = x[i-dimension+1+j]
                else:
                    data[i,:] = x[i]
    return data


# In[5]:


def standardize(x,mean,stdv):
    x = (x-mean)/stdv
    return x

def inverse_standardize(x, mean, stdv):
    x = x*stdv + mean
    return x
def normalization(x, max_x, min_x):
    x = (x-min_x)/(max_x-min_x)
    return x

def denormalization(x,max_x,min_x):
    x = x*(max_x-min_x) + min_x
    return x
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# In[6]:


def train_loop(model, L2_lambda, learning_rate, batch_size, train_loader, valid_loader):
    n_epochs = 5
    loss_function = nn.L1Loss()#nn.MSELoss()--> fonction cout a optimiser
    optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma=0.99)
    #early stopping
    last_loss = 100
    patience = 5
    trigger_times = 0
    train_loss = []
    valid_loss = []
    lrs = []
    
    for epoch in range(n_epochs):
        #y_train.shape[1] = 10 = horizon
        #on doit avoir en output une shape de 11 avec le token END
        #pred_train = np.zeros((len(train_loader)*batch_size,y_train.shape[1]+1))
        pred_train = np.zeros((len(train_loader)*batch_size,y_train.shape[1]-1))
        count = 0
        epoch_loss = []
        for batch in (train_loader):
            x = batch[0]
            y = batch[1]
            #besoin des floats?
            x = x.float()
            y = y.float() 
            """
            target = y
            x_mask = None
            src_inp = x 
            
            eos_token = torch.randn((1,1))
            eos_token = eos_token.expand(batch_size, -1).cuda()
            target = torch.cat((y,eos_token),-1)#la sequence de N mots suivie du token END
            
            sos_token = torch.randn((1,1))
            sos_token = sos_token.expand(batch_size, -1).cuda()
            dec_inp = torch.cat((sos_token, target), 1)      
            dec_inp = dec_inp[:,:-1]#on prend le start token suivi des N mots
            """
            enc_inp = x[:,1:-1,:]
            dec_inp = y[:,:-1]
            target = y[:,1::]
            
            src_mask = None#torch.ones((batch_size, 1,src_inp.shape[1]*src_inp.shape[2])).to(device)
            tgt_mask = subsequent_mask(dec_inp.shape[1]).cuda()
           
            #print(tgt_mask)                
            # Compute prediction and loss
            pred = model(enc_inp,src_mask,dec_inp,tgt_mask)#pred = model(x,x_mask,y,tgt_mask)
            #loss = loss_function(pred[:,:-1], target[:,:-1])
            loss = loss_function(pred, target)  
            #Replaces pow(2.0) with abs() for L1 regularization    
            l2_lambda = L2_lambda
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            #l2_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            
            pred = pred.cpu().detach().numpy()#.flatten()
            pred_train[count*pred.shape[0]:(count+1)*pred.shape[0],:] = pred[:,:]
            count += 1
            epoch_loss = np.append(epoch_loss,loss.item())
            # Gradient backpropagation
            optimizer.zero_grad()#optim.optimizer.zero_grad()
            loss.backward()
            optimizer.step()#optim.step()
            lr = optimizer.param_groups[0]["lr"]#optim.rate()
        scheduler.step()
        lrs.append(lr)
        current_loss = validation(model,valid_loader,loss_function,batch_size)
        valid_loss = np.append(valid_loss, current_loss)
        train_loss = np.append(train_loss, np.mean(epoch_loss))
        if current_loss >= last_loss:
            trigger_times += 1
            print('Epoch [{}/{}], Training loss: {:.4f}, Validation loss: {:.4f}'.format(epoch+1,n_epochs,np.mean(epoch_loss),current_loss))
            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                break
        else:
            print('Epoch [{}/{}], Training loss: {:.4f}, Validation loss: {:.4f}'.format(epoch+1,n_epochs,np.mean(epoch_loss),current_loss))
            trigger_times = 0

        last_loss = current_loss
    return train_loss, valid_loss, pred_train, lrs


# In[7]:


def validation(model, valid_loader, loss_function, batch_size):
    loss_total = 0
    with torch.no_grad():
        for batch in valid_loader:
            x_mask = None
            x = batch[0]
            y = batch[1]
            x = x.float()
            y = y.float()
            """
            target = y
            x_mask = None
            src_inp = x 
            
            eos_token = torch.randn((1,1))
            eos_token = eos_token.expand(batch_size, -1).cuda()
            target = torch.cat((y,eos_token),-1)#la sequence de N mots suivie du token END
            
            sos_token = torch.randn((1,1))
            sos_token = sos_token.expand(batch_size, -1).cuda()
            dec_inp = torch.cat((sos_token, target), 1)      
            dec_inp = dec_inp[:,:-1]#on prend le start token suivi des N mots
            """
            
            enc_inp = x[:,1:-1,:]
            dec_inp = y[:,:-1]
            target = y[:,1::]
            
            src_mask = None#torch.ones((batch_size, 1,src_inp.shape[1]*src_inp.shape[2])).to(device)
            tgt_mask = subsequent_mask(dec_inp.shape[1]).cuda()
            
            # Compute prediction and loss
            pred = model(enc_inp,src_mask,dec_inp,tgt_mask)#pred = model(x,x_mask,y,tgt_mask)
            
            loss = loss_function(pred, target)  
            #loss = loss_function(pred[:,:-1], target[:,:-1]) 
            loss_total += loss.item()
            
    return loss_total/len(valid_loader)


# In[8]:


def test_loop(model,x_t, y_t,batch_size):
    Dataset_test = Mydataset(x_t,y_t)   
    test_loader = DataLoader(Dataset_test, batch_size=batch_size, shuffle=False,drop_last=True)#DataLoader(Dataset_test,shuffle=False,drop_last=True)
    test_loader = DeviceDataLoader(test_loader, device)
    loss_function = nn.L1Loss()#nn.MSELoss()#nn.L1Loss()
    loss_total = 0
    pred_test = np.zeros((y_t.shape[0],y_t.shape[1]-1))
    with torch.no_grad():
        count = 0
        for batch in test_loader:
            x_mask = None
            x = batch[0]
            y = batch[1]  
            x = x.float()
            y = y.float()
            
            """
            target = y
            x_mask = None
            src_inp = x 
            eos_token = torch.randn((1,1))
            eos_token = eos_token.expand(batch_size, -1).cuda()
            target = torch.cat((y,eos_token),-1)#la sequence de N mots suivie du token END
            
            sos_token = torch.randn((1,1))
            sos_token = sos_token.expand(batch_size, -1).cuda()
            dec_inp = torch.cat((sos_token, target), 1)      
            dec_inp = dec_inp[:,:-1]#on prend le start token suivi des N mots
            """
            
            enc_inp = x[:,1:-1,:]
            dec_inp = y[:,:-1]
            target = y[:,1::]
            
            src_mask = None#torch.ones((batch_size, 1,src_inp.shape[1]*src_inp.shape[2])).to(device)
            tgt_mask = subsequent_mask(dec_inp.shape[1]).cuda()
            
            # Compute prediction and loss
            pred = model(enc_inp,src_mask,dec_inp,tgt_mask)#pred = model(x,x_mask,y,tgt_mask)
            loss = loss_function(pred, target)           
            
            loss_total += loss.item()
            pred = pred.cpu().detach().numpy()
            pred_test[count*pred.shape[0]:(count+1)*pred.shape[0],:] = pred[:,:]
            count += 1
    score = loss_total/len(test_loader)
    print(f"Test Error: \n  Avg loss: {score:>8f} \n")
    return pred_test,score


# In[9]:


torch.set_default_dtype(torch.float32)
random_seed = 1
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
device = get_default_device()
print("Device",device)

N_values = 32
Horizon = 11
lr = [5e-4]
#un grand batch permet un entrainement plus rapide mais apporte moins de précision
#un petit batch permet un entrainement moins rapide mais apport plus de précision
#dans le cas de petit dataset, on peut donc se permettre d'utiliser un petit batch 
#Stochastic Gradient Descent (SGD) = Gradient Descent with batch size equal to 1
#Mini batch Gradient Descent = Gradient Descent with batch size bigger than 1 
batch_size = [100]
L2_lambda = [0]
n_neurons = [1024]#256
standard_bool =True#decide si on fait une standardisation des donnees ou bien une normalisation des donnees
score = []#np.zeros(len(n_neurons))
plotFig = True

for i in range(len(L2_lambda)):
    for j in range(len(lr)):
        for k in range(len(batch_size)):
            args = argparse.Namespace()
            args.LAYER = 6 #nombre de couche pour l encodeur et le decodeur
            args.HIDDEN_SIZE = 512#inp_dim[i] #dimension de l input --> embedding 
            args.FF_SIZE =  n_neurons[0]#256#nombre de neurones du MLP
            args.MULTI_HEAD = 8 #nombre d abstraction du MHA
            args.DROPOUT_R = 0
            args.OUTPUT_DIMENSION = Horizon
            args.BATCH_SIZE = batch_size[0]
            args.SEQ_LENGTH = N_values
            training = []
            mean_MSE = []       
            test = []
            
            #creation d une liste d image 
            x_train = signal(10)
            x_test = signal(10) 
            
            y_train = np.zeros((len(x_train),Horizon))
            y_test = np.zeros((len(x_test),Horizon)) 
            if standard_bool:
                mean_x_train = np.mean(x_train)
                stdv_x_train = np.std(x_train)  
                
                mean_x_test = np.mean(x_test)
                stdv_x_test = np.std(x_test)  
                 
                print("mean", mean_x_train, mean_x_test)
                print("stdv", stdv_x_train, stdv_x_test)
                
                y_train = association(standardize(x_train,mean_x_train,stdv_x_train),Horizon-1)
                y_test = association(standardize(x_test,mean_x_test,stdv_x_test),Horizon-1)
            else: 
                max_x_train = np.max(x_train)
                max_x_test = np.max(x_test)
                
                min_x_train = np.min(x_train)
                min_x_test = np.min(x_test)
                
                print("max", max_x_train, max_x_test)
                print("min", min_x_train, min_x_test)
                
                y_train = association(normalization(x_train, max_x_train, min_x_train),Horizon-1)
                y_test = association(normalization(x_test, max_x_test, min_x_test),Horizon-1)
                
            x_train = association(x_train, -N_values)
            x_test = association(x_test, -N_values)
            
            sos_token_train = np.random.normal(size=len(x_train))
            sos_token_test = np.random.normal(size=len(x_test))
            eos_token_train = np.random.normal(size=len(x_train))
            eos_token_test = np.random.normal(size=len(x_test))
            
            x_train = np.insert(x_train, 0, sos_token_train,1)
            x_test = np.insert(x_test, 0, sos_token_test,1)
            x_train = np.insert(x_train, x_train.shape[1], eos_token_train,1)
            x_test = np.insert(x_test, x_test.shape[1], eos_token_test,1)
            
            y_train = np.insert(y_train, 0, sos_token_train,1)
            y_test = np.insert(y_test, 0, sos_token_test,1)
            y_train = np.insert(y_train, y_train.shape[1], eos_token_train,1)
            y_test = np.insert(y_test, y_test.shape[1], eos_token_test,1)
            
            x_train = x_train.reshape(-1, N_values+2, 1)
            x_test = x_test.reshape(-1, N_values+2, 1)
            
            print("Train dataset shape", np.array(x_train).shape)
            print("Test dataset shape", x_test.shape)
            
            net = Net(args)
            net = to_device(net,device)
            # Initialize parameters with Glorot / fan_avg.
            for p in net.parameters():
                if p.dim() > 1:
                    nn.init.kaiming_uniform_(p)#nn.init.xavier_uniform_(p)
            print("End of initialization")
            print("Nombre de neurones: ", n_neurons[0])
            print("Learning rate: ", lr[j])
            print("batch_size: ", batch_size[k])
        
            train_set = Mydataset(x_train,y_train)
            test_set = Mydataset(x_test,y_test)   
            train_set_size = int(len(train_set) * 0.8)
            valid_set_size = len(train_set) - train_set_size
            train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size])
            
            #la longueur du dataloader correspond au nombre d iteration que l on va réaliser
            #un dataset de longueur 100 et batch_size = 10 --> len(dataloader) = 10
            
            train_loader = DataLoader(train_set, batch_size=batch_size[k], shuffle=False, drop_last=True)
            #test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
            valid_loader = DataLoader(valid_set, batch_size=batch_size[k], shuffle=False, drop_last=True)
            train_loader = DeviceDataLoader(train_loader, device)
            valid_loader = DeviceDataLoader(valid_loader, device)
            torch.cuda.empty_cache()
            net.train()
            train_loss, valid_loss,pred_train,lrs = train_loop(net,L2_lambda[i],lr[j], batch_size[k], train_loader, valid_loader)
            #pour que le batch normalisation, dropout, ... soient "inverses" lors de la phase de test
            #c'est la meme chose que net.train(False)
            net.eval()
            pred_test,score_test = test_loop(net, x_test, y_test,1)
            #score[i] = score_test
            #score.append(score_test) 
            
            #denormalisation des data 
            #elles sont en mm 
            if standard_bool: 
                Xpos_true = inverse_standardize(y_test,mean_x_test,stdv_x_test)
                Xpos_pred = inverse_standardize(pred_test,mean_x_test,stdv_x_test)
                
                Xpos_train = inverse_standardize(pred_train,mean_x_train,stdv_x_train)
            else: 
                Xpos_true = denormalization(y_test, max_x_test, min_x_test)
                Xpos_pred = denormalization(pred_test, max_x_test, min_x_test)
            
            if plotFig:
                plt.figure(figsize = (10,4))
                plt.plot(train_loss, 'r')
                plt.plot(valid_loss, 'b')
                plt.title("Learning curve of the Transformer")
                plt.legend(["Training score", "Validation score"])
                plt.xlabel("Epochs")
                plt.ylabel("Score")
                #plt.ylim((0,3))
                
                plt.figure(figsize=(10,4))
                plt.plot(range(len(lrs)),lrs)
                plt.title("PyTorch Learning Rate")
                plt.xlabel("epoch")
                plt.ylabel("learning rate")
                
                plt.figure(figsize=(10,4))
                plt.plot(Xpos_pred[0:200,0])
                plt.plot(Xpos_true[0:200,0])
                plt.title("Prediction t+1 selon l'axe x")
                plt.legend(["Prediction","Ground truth"])
                
                plt.figure(figsize=(10,4))
                plt.plot(Xpos_train[0:200,0])
                #plt.plot(Xpos_true[0:200,0])
                plt.title("Training t+1")
                plt.legend(["Prediction","Ground truth"])

