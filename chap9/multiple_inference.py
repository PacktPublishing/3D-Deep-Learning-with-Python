from inference_unseen_image import inference
from PIL import Image
import numpy as np
import imageio

def create_gif(model_path, image_path, save_path, theta = -0.15, phi = -0.1, tx = 0,
              ty = 0, tz = 0.1, num_of_frames = 5):

    '''
    This function creates sequential gif from an input image
    Inputs:
        model_path _ path to pretrained model
        image_path _ first image of the sequence
        save_path _ path to save new reconstructed gif
        theta _ rotation parameter theta
        phi - rotation parameter phi
        tx _ translation parameter tx
        ty _ translation parameter ty
        tz _ translation parameter tz
        num_of_frames _ number of frames for the gif

    Returns:
        Saved gif with 'num_of_frames' reconstructed images
    '''
    im = inference(model_path, test_image=image_path, theta=theta,
                   phi=phi, tx=tx, ty=ty, tz=tz)
    frames = []
    for i in range(num_of_frames):
        im = Image.fromarray((im * 255).astype(np.uint8))
        frames.append(im)
        im = inference(model_path, im, theta=theta,
                   phi=phi, tx=tx, ty=ty, tz=tz)

    imageio.mimsave(save_path, frames,  duration=1)


if __name__ == '__main__':
    MODEL = './synsin/modelcheckpoints/realestate/zbufferpts.pth'
    IMAGE =  'appartement.JPG'
    SAVE = 'test_gif.gif'
    create_gif(MODEL, IMAGE, SAVE)

