"""
Tools for tracking both the gradients and weights of modules in PyTorch.
"""
import torch
import numpy as np
import io
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

def write_dist_parameters(data: torch.Tensor, layer_name: str, value_type: str, writer: SummaryWriter, epoch_id: int):
    """
    Write the mean and variance of the parameters to TensorBoard.
    """
    mean = data.mean().item()
    var = data.var().item()

    writer.add_scalar(f'{layer_name}/{value_type} Mean', mean, epoch_id)
    writer.add_scalar(f'{layer_name}/{value_type} Var', var, epoch_id)


def write_dist_histogram(data: np.ndarray, layer_name: str, color: str, title: str, xlabel: str, ylabel: str, writer: SummaryWriter, epoch_id: int):
    """
    Write a histogram of the data to TensorBoard.
    """

    plt.figure(1)
    plt.hist(data, bins=50, alpha=0.75, color=color)
    plt.title(f"{layer_name} {title}")
    plt.xlabel(f"{xlabel}")
    plt.ylabel(f"{ylabel}")

    # Show the grid
    plt.grid(True)

    # Save the histogram to a BytesIO buffer
    hist_buf = io.BytesIO()
    plt.savefig(hist_buf, format="png")
    hist_buf.seek(0)

    # Convert the buffer content to an image and then to a NumPy array in HWC format
    hist_image = Image.open(hist_buf)
    hist_image_np = np.array(hist_image)

    # Add the histogram to TensorBoard
    writer.add_image(
        f"{layer_name}/{title}", hist_image_np, epoch_id, dataformats="HWC"
    )

    # Close the buffer
    hist_buf.close()
    plt.close()