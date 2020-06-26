from mxnet.gluon.model_zoo import vision
from mxnet.gluon import nn
import mxnet as mx
from mxnet import gluon
import numpy as np
import mxnet.ndarray as nd
import os 

def get_fsr(num_classes, ctx, kernel_size):
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Conv2D(channels=256, kernel_size=1))
        net.add(nn.BatchNorm())
        net.add(nn.Activation('relu'))
        net.add(nn.Conv2D(channels=512, kernel_size=1))
        net.add(nn.BatchNorm())
        net.add(nn.Activation('relu'))
        net.add(nn.Conv2D(channels=1024, kernel_size=kernel_size))
        net.add(nn.BatchNorm())
        net.add(nn.Activation('relu'))
        net.add(nn.Dense(num_classes, flatten=True))
    net.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), ctx=ctx)

    return net

def broad_multiply(model_output, semantic_region,ctx):
    """ model_output = 1024 x 7 x 7  semantic_region  = 20 x 7 x 7"""
    # print(model_output.shape)
    # print(semantic_region.shape)
    batchsz,a,_,_= semantic_region.shape
    # print(model_output.shape)
    # lc=[]
    ll=[]
    for ii in range(0,batchsz):
        # print(ii)
        # res = nd.zeros((1024,7,7),ctx)
        # ll = []
        for i in range(0,a):
            # print(i)
            tmp = semantic_region[ii][i][:][:]
            # b,c = tmp.shape
            res_tmp=model_output[ii]*(tmp.sum())
            # print(tmp.sum())
            # for j in range(0,b):
            #     # clas=0
            #     for k in range(0,c):
            #         # tt=nd.reshape(mx.nd.array(tmp[j][k]),shape=model_output[ii].shape)
            #         # print(model_output[ii])
            #         # print(model_output[ii]*tmp[j][k])
            #         # print(tmp[j][k])
            #         # clas+=tmp[j][k]
            #         res = res+model_output[ii]*(tmp[j][k])
            #     # print(clas)
            # ll.append(res)
            # print("res==tmp: ", res == res_tmp)
            ll.append(res_tmp)
            # print(len(res.shape))
            # res = nd.zeros((1024,7,7),ctx)
        # lc.append(ll)
    stk = nd.stack(*(ll))
        # lc.append(stk)
    # print((lc.size()))
    # nplc=np.array(lc)
    # print((nplc.shape))
    return stk
def seg_attr(x,l):
    batchsize=x.shape[0]
    y=[]
    for bz in range(batchsize):
        gender = l[bz][2]+l[bz][13]
        hair = l[bz][2]
        sunglass = l[bz][4]
        hat = l[bz][1]
        tshirt_longsleeve_formal = l[bz][5]+l[bz][6]+l[bz][7]+l[bz][14]+l[bz][15]
        shorts_jeans_longpants = l[bz][9]+l[bz][10]+l[bz][16]+l[bz][17]
        skirt = l[bz][12]
        facemask = l[bz][13]
        logo_plaid = tshirt_longsleeve_formal

        # print( x[bz][0].shape,  gender.shape)
        y.append(x[bz][0] * gender)
        y.append(x[bz][1] * hair)
        y.append(x[bz][2] * sunglass)
        y.append(x[bz][3] * hat)
        y.append(x[bz][4] * tshirt_longsleeve_formal)
        y.append(x[bz][5] * tshirt_longsleeve_formal)
        y.append(x[bz][6] * tshirt_longsleeve_formal)
        y.append(x[bz][7] * shorts_jeans_longpants)
        y.append(x[bz][8] * shorts_jeans_longpants)
        y.append(x[bz][9] * shorts_jeans_longpants)
        y.append(x[bz][10] * skirt)
        y.append(x[bz][11] * facemask)
        y.append(x[bz][12] * logo_plaid)
        y.append(x[bz][13] * logo_plaid)
    stk = nd.stack(*(y))
    return stk
class seg_cal(nn.Block):
    def __init__(self, num_classes, lst,ctx,**kwargs):
        super(seg_cal, self).__init__(**kwargs)
        self.avgpool = nn.AvgPool2D((7,7))
        # self.fc1 = nn.Conv2D(channels=1024, kernel_size=1)
        self.fc = nn.Dense(num_classes, flatten=True)
        self.l=lst
        self.ctx=ctx
        # self.fc1.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), ctx=ctx)
        self.fc.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), ctx=ctx)


    def forward(self, x):
        # x=self.fc1(x)
        #print(x.shape)
        #print(self.l.shape)
        out = broad_multiply(x,self.l,self.ctx)
        # print(out.shape)
        out = self.avgpool(out)
        # print(out.shape)
        out2 = self.fc(out)
        # print(out2.shape)
        out3 = nd.softmax(out2)
        out = out2*out3
        # print(out.shape)
        # print(x.shape[0],self.l.shape[1])
        out = out.reshape((x.shape[0],self.l.shape[1],-1))
        # print(out.shape)
        out = nd.sum(out, axis=1) 
        # print(out.shape)
        return out
class seg_cal2(nn.Block):
        def __init__(self,num_classes,lst,ctx,**kwargs):
            super(seg_cal2, self).__init__(**kwargs)
            self.l=nd.array(lst,ctx)
            self.fc = nn.Conv2D(channels=num_classes, kernel_size=1)
            self.fc.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), ctx=ctx)
   
        def forward(self, x):
            x = self.fc(x)
            # print(x.shape)
            # print(self.l.shape)
            out = seg_attr(x,self.l)
            out = out.reshape((x.shape[0],x.shape[1],x.shape[2],-1))
            # print(out.shape)
            # l = nd.array(lst,ctx)
        
            return out
class seg_cal3(nn.Block):
        def __init__(self,num_classes,lst,ctx,**kwargs):
            super(seg_cal3, self).__init__(**kwargs)
            self.l=nd.array(lst,ctx)
            # self.fc = nn.Conv2D(channels=num_classes, kernel_size=1)
            # self.fc.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), ctx=ctx)
   
        def forward(self, x):
            # x = self.fc(x)
            # print(x.shape)
            # print(self.l.shape)
            out = seg_attr(x,self.l)
            out = out.reshape((x.shape[0],x.shape[1],x.shape[2],-1))
            # print(out.shape)
            # l = nd.array(lst,ctx)
        
            return out
def get_fatt(num_classes, stride, ctx):
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Conv2D(channels=512, kernel_size=1))
        net.add(nn.BatchNorm())
        net.add(nn.Activation('relu'))
        net.add(nn.Conv2D(channels=512, kernel_size=3, padding=1))
        net.add(nn.BatchNorm())
        net.add(nn.Activation('relu'))
        # net.add(nn.Conv2D(channels=512, kernel_size=3, padding=1))
        # net.add(nn.BatchNorm())
        # net.add(nn.Activation('relu'))
        net.add(nn.Conv2D(channels=num_classes, kernel_size=1, strides=stride))
    net.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), ctx=ctx)
    return net


def get_conv2D(num_classes, stride, ctx):
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Conv2D(channels=num_classes, kernel_size=1, strides=stride))
        net.add(nn.Activation('sigmoid'))
    net.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), ctx=ctx)
    return net


def getResNet(num_classes, ctx, NoTraining=True):
    resnet = vision.resnet101_v1(pretrained=True, ctx=ctx)

    net = vision.resnet101_v1(classes=num_classes, prefix='resnetv10_')
    x = nd.random.uniform(shape=(1,3,224,224),ctx=ctx)
    # for layer in resnet.features:
    #     x=layer(x)
    #     print(layer.name, x.shape)

    with net.name_scope():
        net.output = nn.Dense(num_classes, flatten=True, in_units=resnet.output._in_units)
        net.output.collect_params().initialize(
            mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), ctx=ctx)
        net.features = resnet.features

    net.collect_params().reset_ctx(ctx)

    inputs = mx.sym.var('data')
    out = net(inputs)
    # sym_json = net(mx.sym.var('data')).tojson()
    # json_file = os.path.join('json', 'sym.json')
    # sym_json.save(json_file)
    # arg_shapes, out_shapes, aux_shapes = out.infer_shape(data=(1,3,224,224))
    # print(arg_shapes, out_shapes)#, aux_shapes )
    # print(net.summary(inputs))
    # print(out) #resnetv10_dense1_fwd
    internals = out.get_internals()
    # print(internals)
    outputs = [internals['resnetv10_batchnorm0_fwd_output'],internals['resnetv10_stage3_activation19_output'], internals['resnetv10_stage3_activation22_output'], internals['resnetv10_stage4_activation2_output'],
               internals['resnetv10_dense1_fwd_output']]
    # seg=internals['resnetv10_stage4_activation2_output']
    # print(seg.shape)
    feat_model = gluon.SymbolBlock(outputs, inputs, params=net.collect_params())
    feat_model._prefix = 'resnetv10_'
    if NoTraining:
        feat_model.collect_params().setattr('grad_req', 'null')
    # output_dict = pickle.load( open( path, "rb" ) )
    # output_tensor = output_dict[y]
    return feat_model

def get_flops_params(model):
    '''
    # use the package mxop(https://github.com/hey-yahei/OpSummary.MXNet)
    # Maybe More Accurate

    from mxop.gluon import count_ops
    op_counter = count_ops(model, input_size=(1,3,224,224))
    return op_counter
    '''

    list_conv_flops = []
    list_conv_params = []
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].shape
        output_channels, output_height, output_width = output[0].shape
        assert self._in_channels % self._kwargs['num_group'] == 0

        kernel_ops = self._kwargs['kernel'][0] * self._kwargs['kernel'][1] * (self._in_channels // self._kwargs['num_group'])
        params = output_channels * kernel_ops
        flops = batch_size * params * output_height * output_width

        list_conv_flops.append(flops)
        list_conv_params.append(params)

    list_dense_flops = []
    list_dense_params = []
    def dense_hook(self, input, output):
        batch_size = input[0].shape[0] if len(input[0].shape) == 2 else 1

        weight_ops = self.weight.shape[0] * self.weight.shape[1]
        #print(self.weight.shape)

        flops = batch_size * weight_ops
        list_dense_flops.append(flops)
        list_conv_params.append(weight_ops)

    def get(net):
        for op in net:
            if isinstance(op, nn.Conv2D):
                op.register_forward_hook(conv_hook)
            if isinstance(op, nn.Dense):
                op.register_forward_hook(dense_hook)
        # for blocks in net.features:
        #     for block in blocks:
        #         if hasattr(block, 'branch_proj'):
        #             for op in block.branch_proj:
        #                 if isinstance(op, nn.Conv2D):
        #                     op.register_forward_hook(conv_hook)
        #                 if isinstance(op, nn.Dense):
        #                     op.register_forward_hook(dense_hook)
        #         for op in block.branch_main:
        #             if isinstance(op, nn.Conv2D):
        #                 op.register_forward_hook(conv_hook)
        #             if isinstance(op, nn.Dense):
        #                 op.register_forward_hook(dense_hook)
        #             if isinstance(op, nn.HybridSequential):
        #                 for OP in op:
        #                     if isinstance(OP, nn.Conv2D):
        #                         OP.register_forward_hook(conv_hook)
        #                     if isinstance(OP, nn.Dense):
        #                         OP.register_forward_hook(dense_hook)
        # for op in net.conv_last:
        #     if isinstance(op, nn.Conv2D):
        #         op.register_forward_hook(conv_hook)
        #     if isinstance(op, nn.Dense):
        #         op.register_forward_hook(dense_hook)
        # for op in net.output:
        #     if isinstance(op, nn.Conv2D):
        #         op.register_forward_hook(conv_hook)
        #     if isinstance(op, nn.Dense):
        #         op.register_forward_hook(dense_hook)
    get(model)
    input = nd.random.uniform(-1, 1, shape=(1, 3, 224, 224), ctx=mx.gpu(0))
    out = model(input)
    total_flops = sum(sum(i) for i in [list_conv_flops, list_dense_flops])
    total_params = sum(sum(i) for i in [list_conv_params, list_dense_params])
    return total_flops, total_params




def getDenseNet(num_classes, ctx):
    densenet = vision.densenet201(pretrained=True, ctx=ctx)

    net = vision.densenet201(classes=num_classes, prefix='densenet0_')
    with net.name_scope():
        net.output = nn.Dense(num_classes, flatten=True)
        net.output.collect_params().initialize(
            mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), ctx=ctx)
        net.features = densenet.features

    net.collect_params().reset_ctx(ctx)

    inputs = mx.sym.var('data')
    out = net(inputs)
    internals = out.get_internals()
    outputs = [internals['densenet0_conv3_fwd_output'], internals['densenet0_stage4_concat15_output'],
               internals['densenet0_dense1_fwd_output']]
    feat_model = gluon.SymbolBlock(outputs, inputs, params=net.collect_params())
    feat_model._prefix = 'densenet0_'

    return feat_model
