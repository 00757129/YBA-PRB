# import
from copy import deepcopy
from models.yolo import Model
import torch
from utils.torch_utils import select_device, is_parallel
import yaml

device = select_device('0', batch_size=1)
# model trained by cfg/training/*.yaml
ckpt = torch.load('last.pt', map_location=device) # change the path to the weight want to be reparameter
# print(ckpt['model'].state_dict().keys())
# reparameterized model in cfg/deploy/*.yaml
model = Model('cfg/training/PRB_Series/yolov7-PRB-3PY_boat.yaml', ch=3, nc=2).to(device) # set model structure 

with open('cfg/training/PRB_Series/yolov7-PRB-3PY_boat.yaml') as f:
    yml = yaml.load(f, Loader=yaml.SafeLoader)
anchors = len(yml['anchors'])

# copy intersect weights
state_dict = ckpt['model'].float().state_dict()
exclude = []
intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
model.load_state_dict(intersect_state_dict, strict=False)
model.names = ckpt['model'].names
model.nc = ckpt['model'].nc
# print(model.state_dict().keys())
# reparametrized YOLOR

print(model.state_dict()['model.111.m.0.bias'].data.shape)
print(model.state_dict()['model.111.ia.0.implicit'].data.shape)
print(state_dict['model.111.ia.0.implicit'].shape)
print(state_dict['model.111.m.0.weight'].shape)
print(state_dict['model.111.m.0.weight'].mul(state_dict['model.111.ia.0.implicit']).sum(1).squeeze().shape)


for i in range((model.nc+5)*anchors):
    model.state_dict()['model.111.m.0.weight'].data[i, :, :, :] *= state_dict['model.111.im.0.implicit'].data[:, i, : :].squeeze()
    model.state_dict()['model.111.m.1.weight'].data[i, :, :, :] *= state_dict['model.111.im.1.implicit'].data[:, i, : :].squeeze()
    model.state_dict()['model.111.m.2.weight'].data[i, :, :, :] *= state_dict['model.111.im.2.implicit'].data[:, i, : :].squeeze()

model.state_dict()['model.111.m.0.bias'].data += state_dict['model.111.m.0.weight'].mul(state_dict['model.111.ia.0.implicit']).sum(1).squeeze()
model.state_dict()['model.111.m.1.bias'].data += state_dict['model.111.m.1.weight'].mul(state_dict['model.111.ia.1.implicit']).sum(1).squeeze()
model.state_dict()['model.111.m.2.bias'].data += state_dict['model.111.m.2.weight'].mul(state_dict['model.111.ia.2.implicit']).sum(1).squeeze()
model.state_dict()['model.111.m.0.bias'].data *= state_dict['model.111.im.0.implicit'].data.squeeze()
model.state_dict()['model.111.m.1.bias'].data *= state_dict['model.111.im.1.implicit'].data.squeeze()
model.state_dict()['model.111.m.2.bias'].data *= state_dict['model.111.im.2.implicit'].data.squeeze()

# model to be saved
ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
        'optimizer': None,
        'training_results': None,
        'epoch': -1}

# save reparameterized model
torch.save(ckpt, 'Rep_yolov7.pt')