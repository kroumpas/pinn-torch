from pathlib import Path
import random

import numpy as np
import torch

from model import Pinn
from data import (
    get_dataset,
    # get_orig_dataset, // This version imports a different dataset - keep in mind that this is not the only change necessary for importing the original ds.
)
from trainer import Trainer


def process_test_result(
    test_data: torch.Tensor,
    loss: float,
    preds: torch.Tensor,
    lambda1: int,
    lambda2: int,
):
    preds = preds.detach().cpu().numpy()
    test_arr = np.array(test_data.data)
    p = test_arr[:, 3]
    u = test_arr[:, 4]
    v = test_arr[:, 5]
    p_pred = preds[:, 0]
    u_pred = preds[:, 1]
    v_pred = preds[:, 2]
    # Print a few true vs predicted values for debugging
    print("Sample of true vs predicted values:")
    for i in range(5):  # Print 5 examples for comparison
        print(f"True u: {u[i]:.4e}, Pred u: {u_pred[i]:.4e}")
        print(f"True v: {v[i]:.4e}, Pred v: {v_pred[i]:.4e}")
        print(f"True p: {p[i]:.4e}, Pred p: {p_pred[i]:.4e}")
    # Error
    # err_u = np.linalg.norm(u - u_pred, 2) / np.linalg.norm(u, 2)
    # err_v = np.linalg.norm(v - v_pred, 2) / np.linalg.norm(v, 2)
    # err_p = np.linalg.norm(p - p_pred, 2) / np.linalg.norm(p, 2)
    # err_u = np.mean(np.abs((u - u_pred) / u))
    # err_v = np.mean(np.abs((v - v_pred) / v))
    # err_p = np.mean(np.abs((p - p_pred) / p))
    epsilon = 1e-8
    err_u = np.mean(np.abs((u - u_pred) / (u + epsilon)))
    err_v = np.mean(np.abs((v - v_pred) / (v + epsilon)))
    err_p = np.mean(np.abs((p - p_pred) / p))

    err_lambda1 = np.abs(lambda1 - 1.0)
    err_lambda2 = np.abs(lambda2 - 0.01) / 0.01

    print(f"Error in u: {err_u:.4e}")
    print(f"Error in v: {err_v:.4e}")
    print(f"Error in pressure: {err_p:.4e}")
    print(f"Error in lambda 1: {err_lambda1:.2f}")
    print(f"Error in lambda 2: {err_lambda2:.2f}")

    # Plot
    pass


def main():
    torch.random.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data
    data_path = Path("../your/path")
    print(f"Loading data from: {data_path}")  # Ensure the correct path is being used.
    train_data, test_data, min_x, max_x = get_dataset(data_path)
    print(f"Train Data: {train_data[:5]}")  # Print a few samples to ensure data is loaded correctly.
    print(f"Test Data: {test_data[:5]}")
    # train_data, test_data, min_x, max_x = get_orig_dataset()

    # Model
    # hidden_dims = [128, 128, 128, 128, 128, 128, 128, 128]
    model = Pinn(min_x, max_x)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # Train
    output_dir = Path(
        r"C:\Users\kroub\Documents\GitHub\project_root\result")
    trainer = Trainer(model, output_dir)
    trainer.train(train_data)

    # Test
    ckpt_dir = trainer.get_last_ckpt_dir()
    trainer.load_ckpt(ckpt_dir)
    outputs = trainer.predict(test_data)

    # Test Result
    lambda1 = trainer.model.lambda1.item()
    lambda2 = trainer.model.lambda2.item()
    print("lambda 1:", lambda1)
    print("lambda 2:", lambda2)
    loss = outputs["loss"]
    preds = outputs["preds"]
    process_test_result(test_data, loss, preds, lambda1, lambda2)


if __name__ == "__main__":
    main()
