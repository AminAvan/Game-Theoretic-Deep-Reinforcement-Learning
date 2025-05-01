import sys

sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")
from absl import app
import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# memory_limit = 4 * 1024
# tf.config.experimental.set_virtual_device_configuration(gpus[0],
#                                                         [tf.config.experimental.VirtualDeviceConfiguration(
#                                                             memory_limit=memory_limit)])
# tf.config.experimental.set_virtual_device_configuration(gpus[1],
#                                                         [tf.config.experimental.VirtualDeviceConfiguration(
#                                                             memory_limit=memory_limit)])

# Get list of GPUs
gpus = tf.config.list_physical_devices('GPU')
memory_limit = 4 * 1024  # 4GB

if len(gpus) >= 2:
    print(f"Configuring two GPUs: {gpus[0]}, {gpus[1]}")
    # Configure first GPU
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
    )
    # Configure second GPU
    tf.config.experimental.set_virtual_device_configuration(
        gpus[1],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
    )
elif len(gpus) == 1:
    print(f"Configuring one GPU: {gpus[0]}")
    # Configure single GPU
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
    )
else:
    print("No GPU detected, running on CPU")
    # Disable GPU devices to ensure CPU execution
    tf.config.set_visible_devices([], 'GPU')


# from Experiment import run_maddpg
from Experiment import run_mad4pg
from Experiment import run_optres_edge
from Experiment import run_optres_local
from Experiment import run_ra
from Experiment import run_ddpg
from Experiment import run_d4pg

if __name__ == '__main__':
    # app.run(run_ddpg.main)
    # app.run(run_d4pg.main)
    # app.run(run_maddpg.main)
    app.run(run_mad4pg.main)
    # app.run(run_optres_local.main)
    # app.run(run_optres_edge.main)
    # app.run(run_ra.main)
