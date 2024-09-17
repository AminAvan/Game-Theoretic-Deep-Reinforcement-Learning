import sys
import os
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# update this path to match your local directory structure
# sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")

from absl import app
import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# memory_limit=4 * 1024
# tf.config.experimental.set_virtual_device_configuration(gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
# tf.config.experimental.set_virtual_device_configuration(gpus[1],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

print("TensorFlow version:", tf.__version__)
print("Running on CPU")

# from Experiment import run_maddpg
import run_mad4pg
# from .run_mad4pg import main as run_mad4pg
# from Experiment import run_optres_edge
# from Experiment import run_optres_local
# from Experiment import run_ra
# from Experiment import run_ddpg
# from Experiment import run_d4pg

if __name__ == '__main__':
    # app.run(run_ddpg.main)
    # app.run(run_d4pg.main)
    # app.run(run_maddpg.main)
     app.run(run_mad4pg.main)
    # app.run(run_optres_local.main)
    # app.run(run_optres_edge.main)
    # app.run(run_ra.main)
    
