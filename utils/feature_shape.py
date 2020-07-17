import torch

def get_feature_shape(model):
    model.cuda()
    input = torch.randn(128, 3, 224, 224).cuda()
    b1, b2, b3, pool, out = model(input)
    feat_maps = [b1, b2, b3, pool, out]
    feat_shapes = [e.size() for e in feat_maps]
    print(feat_shapes)
    return feat_shapes