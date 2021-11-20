import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore import dtype as mstype
from mindspore.dataset.vision import Inter

decode_op = CV.Decode(rgb=True)  # jpeg解码
resize_op = CV.Resize([224, 224], Inter.BICUBIC)  # 图像resize,resize尺寸可以在这里修改
type_cast_op_image = C.TypeCast(mstype.float32)
type_cast_op_label = C.TypeCast(mstype.int32)
hwc2chw_op = CV.HWC2CHW()

def preprocess(dataset, bs=2):
    """
    :brief: process a dataset ,include [decode, resize,shuffle,batch]
    :param dataset:  dataset to be processed
    :param bs:  batch_size
    :return: a processed dataset
    """
    # dataset = dataset.map(operations=[decode_op,  resize_op, hwc2chw_op,type_cast_op_image],
    #                       input_columns=["data"],
    #                       output_columns=["data"])
    dataset = dataset.map(operations=decode_op,input_columns="data")
    dataset = dataset.map(operations=resize_op,input_columns="data")
    dataset = dataset.map(operations=hwc2chw_op,input_columns="data")
    dataset = dataset.map(operations=type_cast_op_image,input_columns="data")
    dataset = dataset.map(operations=[type_cast_op_label],
                          input_columns=["label"])
    dataset = dataset.shuffle(5)
    dataset = dataset.batch(batch_size=bs)

    return dataset
