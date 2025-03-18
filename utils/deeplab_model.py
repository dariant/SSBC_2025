import torchvision

def initialize_model(num_classes, keep_feature_extract=False, use_pretrained=True):
    """ DeepLabV3 pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    """

    if use_pretrained: 
        model_deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet101(weights="DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1", progress=True)
    else: 
        model_deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet101()

    model_deeplabv3.aux_classifier = None
    if keep_feature_extract:
        for param in model_deeplabv3.parameters():
            param.requires_grad = False
    
    model_deeplabv3.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)

    return model_deeplabv3