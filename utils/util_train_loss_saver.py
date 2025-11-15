import numpy as np


def save_obj_traj(training_losses_per_batch, training_iter, save_dir):
    np.savetxt(save_dir + '/obj_train_iter_' + str(training_iter),
               np.array(training_losses_per_batch))
