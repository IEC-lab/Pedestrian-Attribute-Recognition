from models import getResNet, get_fsr,seg_cal3
# from mxnet import nd
from utilities import get_iterators
from attention import attention_net_trainer
from evaluation import evaluate_mAP
from utilities import prettyfloat
import numpy as np
from scipy.special import expit
import mxnet as mx
import argparse
from utilities import SimpleLRScheduler
import cv2
from mxnet import nd, autograd, gluon
import time


def test(args, ctx):
    print('Testing Performance')
    modelpath='org/'
    data_shape = (3, 224, 224)
    _, _, test_iter = get_iterators(args.batch_size, args.num_classes, data_shape)
    test_iter.reset()
    # model = getResNet(args.num_classes, ctx)
    model = gluon.SymbolBlock.imports(modelpath+"joint_new_base_resNet-symbol.json",
    ["data"],
    param_file=modelpath+'joint_new_base_resNet-0021.params',
    ctx=ctx)
    # model.load_params(modelpath+'joint_new_base_resNet-0021.params', ctx=ctx,allow_missing=True)#,ignore_extra=True)
    

    # stage_attentions, stages = {}, {}
    # for stage in range(2, 3):
    #     stage_attentions['stage_' + str(stage)] = attention_net_trainer(_, args.num_classes, args, 1, ctx)
    #     stage_attentions['stage_' + str(stage)][0].load_params(modelpath+'joint_new_fconv_stage_' + str(stage) + '.params', ctx=ctx)
    #     stage_attentions['stage_' + str(stage)][1].load_params(modelpath+'joint_new_fatt_stage_' + str(stage) + '.params', ctx=ctx)
    #     kernel = 14
    #     if stage == 4:  # 4
    #         kernel = 7
    #     stages['stage_' + str(stage)] = get_fsr(args.num_classes, ctx, kernel_size=kernel)
    #     stages['stage_' + str(stage)].load_params(modelpath+'joint_new_fsr_stage_' + str(stage) + '.params', ctx=ctx)

    predicts_test, labels_test = [], []
    

    # name_file = open('/home/lyf/git-repo/imbalanced_learning/wider_records/testing_list.lst')
    # name_file_lst=[]
    # for line in name_file:
    #     nfl='.'+line.split()[-1][20:]
    #     # print(nfl)
    #     name_file_lst.append(nfl)
    # path = "/home/lyf/git-repo/Self-Correction-Human-Parsing/seg/test7.txt"
    # # clspath = "/home/lyf/git-repo/Self-Correction-Human-Parsing/seg/class.txt"
    # output_dict = pickle.load( open( path, "rb" ) )
    # # output_dict_cls = pickle.load( open( clspath, "rb" ) )
    # print('load pickle done.')
    st=time.time()
    cnt=0
    for batch_id, batch in enumerate(test_iter):
        data = batch.data[0].as_in_context(ctx)
        label = batch.label[0].as_in_context(ctx)
        # output_tensor_cls=[]

        # st=batch_id*args.batch_size
        # ed=(batch_id+1)*(args.batch_size)
        # output_tensor_lst = output_dict[name_file_lst[st]][0].cpu().numpy()
        # # output_tensor_cls.append(list(output_dict_cls[name_file_lst[st]][0]))
        # if ed>len(name_file_lst)-1:
        #     # ed=len(name_file_lst)
        #     continue
        # for ii in range(st+1,ed):
        #     cpu_tensor= output_dict[name_file_lst[ii]][0].cpu()
        #     cpu_tensor=cpu_tensor.numpy()
        #     # print(cpu_tensor)
        #     # cpu_tensor_cls= output_dict_cls[name_file_lst[ii]][0]
        #     # print(cpu_tensor_cls)
        #     output_tensor_lst=np.concatenate((output_tensor_lst, cpu_tensor), axis=0)
        #     # output_tensor_cls.append(list(cpu_tensor_cls))
        # seg_cal=seg_cal3(args.num_classes,np.array(output_tensor_lst),ctx)
        

        net_features_stg3_v1, net_features_stg3, net_features_stg4, output = model(data)
        # all_stages = {}
        # for stage in range(2, 3):
        #     if stage == 2:
        #         inp_feats = net_features_stg3_v1
        #     elif stage == 3:
        #         inp_feats = net_features_stg3
        #     else:
        #         inp_feats = net_features_stg4

        #     features = stage_attentions['stage_' + str(stage)][0](inp_feats)
        #     # if stage ==4:
        #     #     features = seg_cal(features)
        #     output_att = stage_attentions['stage_' + str(stage)][1](inp_feats)

        #     temp_f = nd.reshape(output_att, (
        #         output_att.shape[0] * output_att.shape[1], output_att.shape[2] * output_att.shape[3]))
        #     spatial_attention = nd.reshape(nd.softmax(temp_f), (
        #         output_att.shape[0], output_att.shape[1], output_att.shape[2], output_att.shape[3]))

        #     attention_features =   features *spatial_attention
        #     all_stages['stage_' + str(stage)] = stages['stage_' + str(stage)](attention_features)
        predictions =  output.asnumpy()
        # predictions = expit(.25 * (sum(all_stages.values()) + output).asnumpy())
        # predictions = output.asnumpy()
        # print(predictions)
        # print((output_tensor_cls))
        # for c in range (predictions.shape[0]):
        #     if 1 in output_tensor_cls[c] and predictions[c][3]>0.85:
        #         predictions[c][3]=1.0
        #     if 2 in output_tensor_cls[c] and predictions[c][1]>0.85:
        #         predictions[c][1]=1.0
        #     if 4 in output_tensor_cls[c] and predictions[c][2]>0.85:
        #         predictions[c][2]=1.0
        #     if 12 in output_tensor_cls[c] and predictions[c][10]>0.85:
        #         predictions[c][10]=1.0
        # print(predictions)
        # print(output_tensor_cls)
        cnt+=1
        predicts_test.extend(predictions)
        labels_test.extend(label.asnumpy())
        # print(batch_id,np.array(predicts_test))
        # print(batch_id)
        # print('{}:{:.3}'.format('Male',predictions[0][0]))
        # print('{}:{:.3}'.format('Long hair',predictions[0][1]))
        # print('{}:{:.3}'.format('Sunglasses',predictions[0][2]))
        # print('{}:{:.3}'.format('Hat',predictions[0][3]))
        # print('{}:{:.3}'.format('T-shirt',predictions[0][4]))
        # print('{}:{:.3}'.format('Long sleeve',predictions[0][5]))
        # print('{}:{:.3}'.format('Formal',predictions[0][6]))
        # print('{}:{:.3}'.format('Short',predictions[0][7]))
        # print('{}:{:.3}'.format('Jeans',predictions[0][8]))
        # print('{}:{:.3}'.format('Long pants',predictions[0][9]))
        # print('{}:{:.3}'.format('Skirt',predictions[0][10]))
        # print('{}:{:.3}'.format('Face mask',predictions[0][11]))
        # print('{}:{:.3}'.format('Logo',predictions[0][12]))
        # print('{}:{:.3}'.format('Plaid',predictions[0][13]))
    spendt=time.time()-st
    print(spendt,cnt, spendt/cnt)
    test_mAP, test_APs = evaluate_mAP(np.array(labels_test), np.array(predicts_test), testingFlag=True)
    print(test_mAP, list(map(prettyfloat, test_APs)))
def image_to_tensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    tensor = nd.array(img)

    rgb_mean = nd.array([104, 117, 123])
    tensor = (tensor.astype('float32')  - rgb_mean)#/ 255 
    tensor = nd.transpose(tensor, [2, 0, 1])#chw
    tensor = nd.expand_dims(tensor, 0)
    return tensor
def demo(args, ctx):
    modelpath='org/'
    model = gluon.SymbolBlock.imports(modelpath+"joint_new_base_resNet-symbol.json",
    ["data"],
    param_file=modelpath+'joint_new_base_resNet-0021.params',
    ctx=ctx)
    # modelpath='t_lr10/'
    # model = getResNet(args.num_classes, ctx)
    # model.load_params(modelpath+'joint_new_base_resNet.params', ctx=ctx,allow_missing=True)
    input_image = cv2.imread("/home/lyf/bs/imbalanced_learning/imbalanced_learning/test.jpg")
    data=image_to_tensor(input_image)
    print(data)
    data=data.as_in_context(ctx)

    net_features_stg3_v1, net_features_stg3, net_features_stg4, output = model(data)
    print(output)
    print('---------------------------------')
    _range = np.max(output) - np.min(output)
    predictions= (output - np.min(output)) / _range
    predictions=predictions.asnumpy().tolist()
    # predictions = output.asnumpy()
    print('{}:{:.3}'.format('Male',predictions[0][0]))
    print('{}:{:.3}'.format('Long hair',predictions[0][1]))
    print('{}:{:.3}'.format('Sunglasses',predictions[0][2]))
    print('{}:{:.3}'.format('Hat',predictions[0][3]))
    print('{}:{:.3}'.format('T-shirt',predictions[0][4]))
    print('{}:{:.3}'.format('Long sleeve',predictions[0][5]))
    print('{}:{:.3}'.format('Formal',predictions[0][6]))
    print('{}:{:.3}'.format('Short',predictions[0][7]))
    print('{}:{:.3}'.format('Jeans',predictions[0][8]))
    print('{}:{:.3}'.format('Long pants',predictions[0][9]))
    print('{}:{:.3}'.format('Skirt',predictions[0][10]))
    print('{}:{:.3}'.format('Face mask',predictions[0][11]))
    print('{}:{:.3}'.format('Logo',predictions[0][12]))
    print('{}:{:.3}'.format('Plaid',predictions[0][13]))
    print(predictions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deep Imbalanced Classification')
    parser.add_argument('--data_path', help='data directory')
    parser.add_argument('--epochs', default=250, type=int, help='epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--wd', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--mom', default=0.9, type=float, help='momentum')
    parser.add_argument('--batch_size', default=24, type=int, help='batch size')
    parser.add_argument('--num_classes', default=14, type=int, help='number of classes')
    parser.add_argument('--finetune', action='store_true', help='fine tune backbone architecture or not?')
    parser.add_argument('--test', action='store_true', help='testing')

    args = parser.parse_args()

    # Parameter Naming
    # params_name = 'saved_models/base_resNet.params'

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ctx = mx.gpu()
    demo(args, ctx)
