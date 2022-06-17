from inference_unseen_image import inference
from PIL import Image
import numpy as np
import imageio

def create_gif(model_path, image_path, save_path, theta = -0.15, phi = -0.1, tx = 0,
              ty = 0, tz = 0.1, num_of_frames = 5):
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

