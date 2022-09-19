## This code is adapted from
## More information about SMPL is available here: https://smpl.is.tue.mpg.de/index.html
from absl import app
from absl import flags

from serialization import load_model
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string("model_file", None, "Path to the model file")

def main(argv):
    del argv
    ## Load SMPL model using the path specified in the flags
    m = load_model(FLAGS.model_file)

    ## Assign random pose and shape parameters
    m.pose[:] = np.random.rand(m.pose.size) * .2
    m.betas[:] = np.random.rand(m.betas.size) * .03

    ## Write to an .obj file
    outmesh_path = './hello_smpl.obj'
    with open( outmesh_path, 'w') as fp:
        for v in m.r:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

        for f in m.f+1: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

    ## Print message
    print('Output mesh {} saved to: {}'.format(outmesh_path, outmesh_path))

if __name__ == '__main__':
    app.run(main)
