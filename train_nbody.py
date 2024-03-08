import argparse
import torch
from torch_geometric.loader import DataLoader

from datasets.nbody_dataset import NBodyDataset, get_nbody_datasets
from nets.equiformer_v2.equiformer_v2_nbody import EquiformerV2_nbody
from engine import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser("Training EquiformerV2 on N-body", add_help=False)
    parser.add_argument("--data-path", type=str, default="datasets/nbody")
    parser.add_argument(
        "--dataset-arg", type=str, default="train_charged5_initvel1tiny"
    )
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_dataset, val_dataset, test_dataset = get_nbody_datasets(
        root=args.data_path,
        dataset_arg=args.dataset_arg,
        train_size=800,
        val_size=100,
        test_size=100,
        seed=42,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize model
    model = (
        EquiformerV2_nbody()
    )  # Update irreps_in and irreps_out based on your dataset
    model.to(device)

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = torch.nn.MSELoss()

    # Training loop
    for epoch in range(args.num_epochs):
        train_stats = train_one_epoch(
            model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            criterion=criterion,
        )
        print(f"Epoch: {epoch}, Train loss: {train_stats['loss']:.4f}")

        if (epoch + 1) % 10 == 0:
            val_stats = evaluate(model, val_loader, device)
            print(f"Epoch: {epoch}, Val loss: {val_stats['loss']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "EquiformerV2 training script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
