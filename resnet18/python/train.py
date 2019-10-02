import torch
import torchvision
import numpy as np
import cv2

def preprocess_image(img):
	
	img = cv2.resize(img,dsize = (224,224))
	img = img[:,:,::-1] # BGR -> RGB
	img = np.transpose(img, (2,0,1))
	img = np.expand_dims(img,0)
	img = img/255.0
		
	return img

def read_classes_names(filepath):

	classes = []
	with open(filepath, 'r') as f:
		lines = f.read().splitlines()
		for line in lines:
			idx, name = line.split(':')
			idx = int(idx)
			name = name[2: -2]
			classes.append(name.lower())
			
	return classes

if __name__ == "__main__":
	
	model = torchvision.models.resnet18(pretrained=True)
	model.eval()
	
	img = cv2.imread('/tmp/python/tiger.jpg')
	img = preprocess_image(img)
	tensor_img = torch.FloatTensor(img)
	logits = model(tensor_img)
	probs =torch.nn.functional.softmax(logits, dim=-1)

	probs, idxs = torch.topk(probs[0], k=5)
	idxs = idxs.data.numpy().astype(np.int32)
	probs = probs.data.numpy()
	
	classes = read_classes_names('/tmp/python/imagenet_names.txt')
	for prob, idx in zip(probs, idxs):
		print ('--class:', classes[idx], '--probability:', prob)

	example_input = torch.rand(1, 3, 224, 224)
	traced_script_module = torch.jit.trace(model, example_input)
	traced_script_module.save("/tmp/python/model.pt")
