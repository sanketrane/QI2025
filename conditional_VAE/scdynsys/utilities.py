import torch
import numpy as np
import tqdm
from pyro.infer import SVI



cell_type_colors = {
    'T1': '#D32F2F',
    'T2': '#FF9800',
    'MZ': '#2196F3',
    'FM': '#9C27B0'
}


def train_loop_full_dataset(
    svi: SVI,
    xs_raw: np.array,
    xtime_raw: np.array,
    utime_raw: np.array,
    num_epochs: int,
    device: torch.device
) -> tuple[int, float]:
    """
    Train the model using the full dataset.
    Parameters
    ----------
    svi : SVI
        The SVI object for training.
    xs_raw : np.array
        The raw input data.
    xtime_raw : np.array
        The time indices for each data point.
    utime_raw : np.array
        The actual time values for each data point.
    num_epochs : int
        The number of epochs to train.
    device : torch.device
        The device to use for training (CPU or GPU).
    Returns
    -------
    train_elbo : list of tuple[int, float]
        A list of tuples containing epoch number and training ELBO value.
    
    """

    # set up the progress bar
    trange = tqdm.trange # show a progress bar
    
    # move data to the correct device
    xs_tensor = torch.tensor(xs_raw, device=device, dtype=torch.float32)
    xtime_tensor = torch.tensor(xtime_raw, device=device, dtype=torch.long)
    utime_tensor = torch.tensor(utime_raw, device=device, dtype=torch.float32)

    # The model and guide functions expect the number of samples as input
    N = torch.tensor(xs_tensor.shape[0], device=device, dtype=torch.float32)
    
    # Initialize a list to store the training ELBO values for plotting
    train_elbo = []
            
    # training loop
    for epoch in (pbar := trange(num_epochs)):
        # perform a single SVI step and get the loss
        total_epoch_loss_train = svi.step(xs_tensor, xtime_tensor, utime_tensor, N)
        
        # append the loss to the list
        train_elbo.append((epoch, total_epoch_loss_train))
        if epoch % 10 == 0:
            pbar.set_description(f"train loss: {total_epoch_loss_train:0.2f}")

    return train_elbo