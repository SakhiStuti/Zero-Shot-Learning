import pickle
import os
import numpy as np
import torchfile
import glob
from torch.autograd import Variable
import torch

def getTrain_Img_Desc_Att(img_path, desc_path, att_path):
    all_files = glob.glob(img_path + '/**.t7', recursive=True)
    assert len(all_files) > 0
    print('Number of files in directory: {}'.format(len(all_files)))
    with open(att_path, 'rb') as handle:
        att_dict =  pickle.load(handle)
        
    for i,file in enumerate(all_files):
        class_name = os.path.basename(file)
        c_name = class_name[:3]
        #Load image, desc and attribute files
        temp_img = torchfile.load(file, force_8bytes_long = True)
        desc_file = os.path.join(desc_path,class_name)
        temp_desc = torchfile.load(desc_file, force_8bytes_long = True)
        temp_att = np.expand_dims(np.array(att_dict[c_name]),axis = 0)
        if i ==0 :
            img = temp_img.transpose(0,2,1).reshape(temp_img.shape[0]*temp_img.shape[2],temp_img.shape[1])
            desc = temp_desc.transpose(0,2,1).reshape(temp_desc.shape[0]*temp_desc.shape[2], temp_desc.shape[1])
            att = np.repeat(temp_att, img.shape[0], axis = 0)
        else:
            img = np.append(img, temp_img.transpose(0,2,1).reshape(temp_img.shape[0]*temp_img.shape[2],temp_img.shape[1]), axis=0)
            desc = np.append(desc, temp_desc.transpose(0,2,1).reshape(temp_desc.shape[0]*temp_desc.shape[2], temp_desc.shape[1]), axis=0 )
            att = np.append(att,np.repeat(temp_att, temp_img.shape[0]*temp_img.shape[2], axis = 0) , axis = 0)
        assert (img.shape[0] == desc.shape[0]), 'Shape not match for class {}, img {} and desc {}'.format(class_name, img.shape,desc.shape)
        assert (img.shape[0] == att.shape[0])
    print('Total training images: {}'.format(img.shape[0]))
    return img,desc,att


def getTest_Img_proto_labels(img_path, desc_path, att_path):
    all_files = glob.glob(img_path + '/**.t7')
    print('Number of files in directory: {}'.format(len(all_files)))
    with open(att_path, 'rb') as handle:
        att_dict =  pickle.load(handle)
        
    for i,file in enumerate(all_files):
        class_name = os.path.basename(file)
        c_name = class_name[:3]
        #Load image, desc and attribute files
        temp_img = torchfile.load(file, force_8bytes_long = True)
        desc_file = os.path.join(desc_path,class_name)
        temp_desc = torchfile.load(desc_file, force_8bytes_long = True)
        temp_att = np.expand_dims(np.array(att_dict[c_name], dtype = 'float32'),axis = 0)
        if i ==0 :
            img = temp_img.transpose(0,2,1).reshape(temp_img.shape[0]*temp_img.shape[2],temp_img.shape[1])
            desc = [temp_desc.transpose(0,2,1).reshape(temp_desc.shape[0]*temp_desc.shape[2], temp_desc.shape[1])]
            att = [np.repeat(temp_att, img.shape[0], axis = 0)]
            proto2label = [i]
            x2label = [i]*(temp_img.shape[0]*temp_img.shape[2])
        else:
            img = np.append(img, temp_img.transpose(0,2,1).reshape(temp_img.shape[0]*temp_img.shape[2],temp_img.shape[1]), axis=0)
            desc.append(temp_desc.transpose(0,2,1).reshape(temp_desc.shape[0]*temp_desc.shape[2], temp_desc.shape[1]))
            att.append(np.repeat(temp_att, temp_img.shape[0]*temp_img.shape[2], axis = 0))
            proto2label += [i]
            x2label += [i]*(temp_img.shape[0]*temp_img.shape[2])
        assert (len(x2label) == img.shape[0]), 'Shape not match for class {}, img {} and desc {}'.format(class_name, img.shape,len(desc))
    assert (len(proto2label)==50), 'Mistake in finding proto2label'
    print('Total test images: {}'.format(img.shape[0]))
    return img,desc,att, x2label, proto2label


def data_iterator(train_x, train_desc, train_att, device):
	""" A simple data iterator """
	batch_idx = 0
	while True:
		# shuffle labels and features
		idxs = np.arange(0, len(train_x))
		np.random.shuffle(idxs)
		shuf_visual = train_x[idxs]
		shuf_desc = train_desc[idxs]
		shuf_att = train_att[idxs] 
		batch_size = 100

		for batch_idx in range(0, len(train_x), batch_size):
			visual_batch = shuf_visual[batch_idx:batch_idx + batch_size]
			visual_batch = visual_batch.astype("float32")
			visual_batch = Variable(torch.from_numpy(visual_batch).float().to(device))
			att_batch = shuf_att[batch_idx:batch_idx + batch_size]
			att_batch = Variable(torch.from_numpy(att_batch.astype("float32")).float().to(device))
			desc_batch = shuf_desc[batch_idx:batch_idx + batch_size]
			desc_batch = Variable(torch.from_numpy(desc_batch).to(torch.long).to(device))			
			#print('qwert:',att_batch.type())
			
			yield att_batch, visual_batch, desc_batch