import torch


class HyperParameters:
    BUFFER_SIZE = int(1e5)
    BATCH_SIZE = 64
    GAMMA = 0.99
    TAU = 1e-3
    LR = 5e-4
    UPDATE_EVERY = 4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
