import logging

import torch

logging.basicConfig(level=logging.INFO)


def get_preferred_device():
    """
    Determine and log the best available device for PyTorch operations based on a priority order: CUDA, MPS, CPU.
    Includes detailed reasoning when MPS is not available.

    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        logging.info("Using CUDA device.")
        return "cuda"

    if torch.backends.mps.is_available():
        logging.info("Using MPS device.")
        return "mps"
    else:
        if not torch.backends.mps.is_built():
            logging.info(
                "MPS not available because the current PyTorch install was not built with MPS enabled."
            )
        else:
            logging.info(
                "MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine."
            )

    logging.info("Using CPU device.")
    return "cpu"


if __name__ == "__main__":
    device = get_preferred_device()
    logging.info("Selected device: %s", device)
