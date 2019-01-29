from models.multi_lenet import MultiLeNetO, MultiLeNetR
from models.segnet import SegnetEncoder, SegnetInstanceDecoder, SegnetSegmentationDecoder, SegnetDepthDecoder
from models.pspnet import SegmentationDecoder, get_segmentation_encoder
from models.multi_faces_resnet import ResNet, FaceAttributeDecoder, BasicBlock
import torchvision.models as model_collection


def get_model(params):
    data = params['dataset']
    if 'mnist' in data:
        model = {}
        model['rep'] = MultiLeNetR()
        model['rep'].cuda()
        if 'L' in params['tasks']:
            model['L'] = MultiLeNetO()
            model['L'].cuda()
        if 'R' in params['tasks']:
            model['R'] = MultiLeNetO()
            model['R'].cuda()
        return model

    if 'cityscapes' in data:
        model = {}
        model['rep'] = get_segmentation_encoder() # SegnetEncoder()
        #vgg16 = model_collection.vgg16(pretrained=True)
        #model['rep'].init_vgg16_params(vgg16)
        model['rep'].cuda()
        if 'S' in params['tasks']:
            model['S'] = SegmentationDecoder(num_class=19, task_type='C')
            model['S'].cuda()
        if 'I' in params['tasks']:
            model['I'] = SegmentationDecoder(num_class=2, task_type='R')
            model['I'].cuda()
        if 'D' in params['tasks']:
            model['D'] = SegmentationDecoder(num_class=1, task_type='R')
            model['D'].cuda()
        return model

    if 'celeba' in data:
        model = {}
        model['rep'] = ResNet(BasicBlock, [2,2,2,2])
        model['rep'].cuda()
        for t in params['tasks']:
            model[t] = FaceAttributeDecoder()
            model[t].cuda()
        return model

