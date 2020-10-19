import argparse
import torch
import os, glob
from image import load_img_to_tensor
from model import load_torch_model, decode_model_prediction



inputImage="http://www" # will come from user



def main():

    #DOWNLOAD INPUTiMAGE -->IMAGES/
    if os.path.isdir(args.input_path):
        fp = glob.glob(os.path.join(args.input_path, '**.jpg'))
        img_tensor = load_img_to_tensor(fp, im_size=(256, 256))
    else:
        img_tensor = load_img_to_tensor(args.input_path, im_size=(256, 256))
        landmark_tensor = torch.zeros(8)

        print('\n50 Category And 30 Attribute Fine Prediction!')
        model = load_torch_model('models/category_attribute_fine.pth', evaluate=True)
        attr_prob, cate_prob = model(img_tensor, landmark=landmark_tensor, return_loss=False)
        if attr_prob.shape[0] != 1 and cate_prob.shape[0] != 1:
            for sample, path in zip(range(attr_prob.shape[0]), fp):
                print(f'--------\tImage in {path}:')
                decode_model_prediction('fine', attr_prob[sample], 'attr')
                decode_model_prediction('fine', cate_prob[sample], 'cate')
        else:
            decode_model_prediction('fine', attr_prob, 'attr')
            decode_model_prediction('fine', cate_prob, 'cate')

        print('\n1000 Attribute Coarse Prediction!')
        model = load_torch_model('models/attribute_coarse.pth', evaluate=True)
        attr_prob = model(img_tensor)['loss_attr']
        if attr_prob.shape[0] != 1:
            for sample, path in zip(range(attr_prob.shape[0]), fp):
                print(f'--------\tImage in {path}:')
                decode_model_prediction('coarse', attr_prob[sample], 'attr')
        else:
            decode_model_prediction('coarse', attr_prob, 'attr')

    #REMOVE IMAGE FROM IMAGES
if __name__ == '__main__':
    main()
