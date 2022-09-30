import cv2
import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl.serialization import load_model

## Load SMPL model (here we load the neural model)
m = load_model('../smplify/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

## Assign random pose and shape parameters
m.pose[:] = np.random.rand(m.pose.size) * .2
m.betas[:] = np.random.rand(m.betas.size) * .03
m.pose[0] = np.pi

## Create OpenDR renderer
rn = ColoredRenderer()

## Assign attributes to renderer
w, h = (640, 480)

rn.camera = ProjectPoints(v=m, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w,w])/2., c=np.array([w,h])/2., k=np.zeros(5))
rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
rn.set(v=m, f=m.f, bgcolor=np.zeros(3))

## Construct point light source
rn.vc = LambertianPointLight(
    f=m.f,
    v=rn.v,
    num_verts=len(m),
    light_pos=np.array([-1000,-1000,-2000]),
    vc=np.ones_like(m)*.9,
    light_color=np.array([1., 1., 1.]))


cv2.imshow('render_SMPL', rn.r)
cv2.waitKey(0)
cv2.destroyAllWindows()