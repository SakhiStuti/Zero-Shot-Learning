import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from utils import process_dict, getEmbed_layer
from dataloader import getTrain_Img_Desc_Att, getTest_Img_proto_labels, data_iterator
import torchfile
import kNN

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim =50
num_layers=2

def compute_accuracy(test_proto_visual, test_visual, test_proto2label, test_x2label):
	# test_proto: [50, 312]
	# test visual: [2993, 1024]
	# test_proto2label: proto2label [50]
	# test_x2label: x2label [2993]
	outpred = [0] * test_visual.shape[0]
	# test_visual: [2993, 1024],
	# test_2label : [2993]

	for i in range(test_visual.shape[0]):
		outputLabel = kNN.kNNClassify(test_visual[i, :], test_proto_visual.cpu().data.numpy(), test_proto2label, 1)
		outpred[i] = outputLabel
	outpred = np.array(outpred)
	acc = np.equal(outpred, test_x2label).mean()
	#print(outpred.dtype,test_x2label.dtype)
	return acc

class MahalanobisLayer(nn.Module):

    def __init__(self, dim, decay = 0.1):
        super(MahalanobisLayer, self).__init__()
        self.register_buffer('S', torch.eye(dim))
        self.register_buffer('S_inv', torch.eye(dim))
        self.decay = decay

    def forward(self, x, x_fit):
        """
        Calculates the squared Mahalanobis distance between x and x_fit
        """

        delta = x - x_fit
        m = torch.mm(torch.mm(delta, self.S_inv), delta.t())
        return torch.diag(m)

    def cov(self, x):
        x -= torch.mean(x, dim=0)
        return 1 / (x.size(0) - 1) * x.t().mm(x)

    def update(self, X, X_fit):
        delta = X - X_fit
        self.S = (1 - self.decay) * self.S + self.decay * self.cov(delta)
        try:
            self.S_inv = torch.pinverse(self.S)
        except:# expection as e:
            #print('Exception in calculaitng inverse... trying a way around')
            S = (1 - self.decay) * self.S.cpu().numpy() + self.decay * np.cov(delta.cpu().numpy(), rowvar=False)
            self.S = torch.from_numpy(S).float().cuda()
            S_inv = np.linalg.inv(S)
            self.S_inv = torch.from_numpy(S_inv).float().cuda()
            #print('Way Around Worked!!')


class Semantic_model(nn.Module):
    def __init__(self, num_layers=1, input_dim=50, hidden_dim=256, output_dim=1024, batchfirst=True):
        super(Semantic_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_dim,hidden_size=hidden_dim, 
                            bidirectional=True, num_layers=num_layers, batch_first=True, dropout= 0.2)
        self.__init_weights()
    
    def __init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, x):
        out , _ = self.lstm(x)
        out_avg = torch.mean(out, dim=1, keepdim=False)
        return out_avg

net = Semantic_model(input_dim=input_dim, num_layers=num_layers).to(device)



#train visual features and description
train_x, train_desc, train_att = getTrain_Img_Desc_Att(
    '../images/train', '../word_c10/train', '../att_per_classes.npy')

iter_ = data_iterator(train_x, train_desc, train_att, device)

#test visual features
val_x, val_pre_proto, val_pre_att_proto,val_x2label, val_proto2label = getTest_Img_proto_labels(
    '../images/val', '../word_c10/val', '../att_per_classes.npy') 


#NOTE: Here Vocab starts from 1 where 1 being '<END>'
vocab = torchfile.load('../vocab_c10.t7', force_8bytes_long = True)
word2idx_vocab = {str(key, 'utf-8'):value for key,value in vocab.items()}
print('Vocabs Loaded: Total words in vocabulary including pad....{}'.format(len(word2idx_vocab)))
#Gives all words in glove's vocab to its glove embedding matrix (dictionary contains 0 as'<PAD>')
glove_dict = process_dict('../glove.6B.50d.txt', dim=input_dim) 

print('Loading embedding layer....')
embed_layer = getEmbed_layer(word2idx=word2idx_vocab, glove_dict=glove_dict,dim=input_dim).to(device)
print('Loaded Embedding!!')

w1 = Variable(torch.FloatTensor(512, 700).to(device), requires_grad=True)
b1 = Variable(torch.FloatTensor(700).to(device), requires_grad=True)
w2 = Variable(torch.FloatTensor(700, 1024).to(device), requires_grad=True)
b2 = Variable(torch.FloatTensor(1024).to(device), requires_grad=True)
w3 = Variable(torch.FloatTensor(312, 700).to(device), requires_grad=True)
b3 = Variable(torch.FloatTensor(700).to(device), requires_grad=True)

#Initialize Mahanabolis
mahalanobis_layer = MahalanobisLayer(1024,decay=0.001).to(device)

# must initialize!
w1.data.normal_(0, 0.02)
w2.data.normal_(0, 0.02)
w3.data.normal_(0, 0.02)
b1.data.fill_(0)
b2.data.fill_(0)
b3.data.fill_(0)



def forward(desc, att,visual_batch_val, update_mode=False):#desc: (N,30) att: (N,313)
    desc = embed_layer(desc) # x: (N,30,glove_dim)
    desc = net(desc) #lstm output (N,512)
    desc = F.relu(torch.mm(desc, w1) + b1)
    att = F.relu(torch.mm(att, w3) + b3)
    
    fusion = 0.5*desc + 0.5*(att)
    fusion = F.relu(torch.mm(fusion, w2) + b2)
    if not update_mode:
        fusion= mahalanobis_layer(visual_batch_val,fusion)
    
    return fusion


criterion = nn.L1Loss()
optimizer = torch.optim.Adam([w1, b1, w2, b2, w3, b3] + list(net.parameters()) + list(embed_layer.parameters()), lr=0.0005, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [2000], gamma=0.1)
best_accuracy = 0.00000

losses,accuracy = [],[]

for i in range(1000000):
    #scheduler.step()
    net.train()
    att_batch_val, visual_batch_val, desc_batch_val = next(iter_)
    pred = forward(desc_batch_val, att_batch_val,visual_batch_val)
    y_batch = torch.zeros_like(pred) 
    loss = criterion(pred, y_batch)

    optimizer.zero_grad()
    loss.backward()
    losses.append(loss.item())
    torch.nn.utils.clip_grad_norm([w1, b1, w2, b2, w3, b3], 1)
    optimizer.step()
    with torch.no_grad():
       X_fit = forward(desc_batch_val, att_batch_val, visual_batch_val, update_mode=True)
       mahalanobis_layer.update(visual_batch_val, X_fit)
    if i % 50 == 0:
        print(loss.item())
        val_proto_visual = torch.zeros((50,1024))
        net.eval()
        for x,proto in enumerate(val_pre_proto):
            proto_desc = torch.LongTensor(proto).to(device)
            proto_att = torch.FloatTensor(val_pre_att_proto[x]).to(device)
            val_proto_visual[x,:] = torch.mean(forward(proto_desc, proto_att,visual_batch_val,update_mode=True), dim=0)
        val_proto_visual = val_proto_visual.to(device)
        dummy = np.array([1]*50)[:,np.newaxis]
        outpred = [0]*val_x.shape[0]
        for i in range(val_x.shape[0]):
            
            #Get the ith example
            curr_val = val_x[i,:][np.newaxis,:]
            curr_val_matrix = torch.FloatTensor(np.dot(dummy, curr_val)).to(device) #create [50,1024] matrix of all with same rows
            #calculate the distance
            dists = mahalanobis_layer(val_proto_visual,curr_val_matrix)
            #sort and assign to lowest dist's label
            _, sortedIndices = torch.sort(dists)
            outpred[i] = val_proto2label[sortedIndices[0].item()]
        
        #Calculate accuracy
        outpred = np.array(outpred)
        acc = np.equal(outpred, val_x2label).mean()
        accuracy.append(acc)      
        if (acc.item() > best_accuracy):
            print('New Best Accuracy: {}'.format(acc))
            net_params = dict(net.state_dict())
            net_params.update(embed_layer.state_dict())
            net_params.update({'w1':w1, 'w2':w2, 'w3':w3, 'b1':b1, 'b2':b2, 'b3':b3})
            torch.save(net_params, './model/fusion_mah/weights_{}.pth'.format(i))
            best_accuracy = acc
        print('Accuracy: {}'.format(acc))
          
