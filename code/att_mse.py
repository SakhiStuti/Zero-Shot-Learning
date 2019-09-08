import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from utils import process_dict, getEmbed_layer
from dataloader import getTrain_Img_Desc_Att, getTest_Img_proto_labels, data_iterator
import torchfile
import kNN

def compute_accuracy(test_proto_visual, test_visual, test_proto2label, test_x2label):
	# test_proto: [50, 312]
	# test visual: [2993, 1024]
	# test_proto2label: proto2label [50]
	# test_x2label: x2label [2993]
	outpred = [0] * test_visual.shape[0]
	for i in range(test_visual.shape[0]):
		outputLabel = kNN.kNNClassify(test_visual[i, :], test_proto_visual.cpu().data.numpy(), test_proto2label, 1)
		outpred[i] = outputLabel
	outpred = np.array(outpred)
	acc = np.equal(outpred, test_x2label).mean()
	return acc

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim =50
num_layers=2



#train visual features and description
train_x, train_desc, train_att = getTrain_Img_Desc_Att(
    '../images/train', '../word_c10/train', '../att_per_classes.npy')


#train iterator
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

w1 = Variable(torch.FloatTensor(312, 700).to(device), requires_grad=True)
b1 = Variable(torch.FloatTensor(700).to(device), requires_grad=True)
w2 = Variable(torch.FloatTensor(700, 1024).to(device), requires_grad=True)
b2 = Variable(torch.FloatTensor(1024).to(device), requires_grad=True)

# must initialize!
w1.data.normal_(0, 0.02)
w2.data.normal_(0, 0.02)
b1.data.fill_(0)
b2.data.fill_(0)


def forward(desc, att):#desc: (N,30) att: (N,313)
    
    a1 = F.relu(torch.mm(att, w1) + b1)
    a2 = F.relu(torch.mm(a1, w2) + b2)
    return a2


def getloss(pred, x):
	loss = torch.pow(x - pred, 2).sum()
	loss /= x.size(0)
	return loss

optimizer = torch.optim.Adam([w1, b1, w2, b2], lr=0.0005, weight_decay=1e-2)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15000, gamma=0.05)

best_accuracy = 0.00000

losses,accuracy = [],[]

for i in range(1000000):
    #scheduler.step()
    att_batch_val, visual_batch_val, desc_batch_val = next(iter_)
    pred = forward(desc_batch_val, att_batch_val)
    loss = getloss(pred, visual_batch_val)

    optimizer.zero_grad()
    loss.backward()
    losses.append(loss.item())
    
    torch.nn.utils.clip_grad_norm([w1, b1, w2, b2], 1)
    optimizer.step()
    
    if i % 500 == 0:
        print(loss.item())
        val_proto_visual = torch.zeros((50,1024))
        for x,proto in enumerate(val_pre_proto):
            proto_desc = torch.LongTensor(proto).to(device)
            proto_att = torch.FloatTensor(val_pre_att_proto[x]).to(device)
            val_proto_visual[x,:] = torch.mean(forward(proto_desc, proto_att), dim=0)

        acc  = compute_accuracy(val_proto_visual, val_x, np.array(val_proto2label), np.array(val_x2label))
        accuracy.append(acc)
        
 
        if (acc.item() > best_accuracy):
            print('New Best Accuracy: {}'.format(acc))
            net_params = {'w1':w1, 'w2':w2, 'b1':b1, 'b2':b2}
            torch.save(net_params, './model/att_mse/weights_{}.pth'.format(i))
            best_accuracy = acc
        
        print('Iterations {} Accuracy: {}'.format(i,acc))
          
