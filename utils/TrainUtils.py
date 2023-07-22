from tqdm import tqdm
import numpy as np
import torch


def train_one_epoch(model, optimizer, train_loader, device, criterion, metric):
    loss_step, metric_step = [], []

    model.train()
    for (inp_data, labels) in train_loader:
        # Move imgs and labels to gpu
        labels = labels.to(device).squeeze()
        inp_data = inp_data.to(device)

        # Forward pass
        outputs = model(inp_data)
        loss = criterion(outputs, labels)

        # Compute gradients and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Compute and record metric and loss
        metric_step.append(metric(outputs, labels).item())
        loss_step.append(loss.item())

    loss_curr_epoch = np.mean(loss_step)
    train_metric = np.mean(metric_step)

    return loss_curr_epoch, train_metric


def validate(model, val_loader, criterion, metric, args):
    loss_step, metric_step = [], []

    model.eval()

    with torch.no_grad():
        for inp_data, labels in val_loader:
            # Move imgs and labels to gpu
            labels = labels.to(args.device).squeeze()
            inp_data = inp_data.to(args.device)
            # Forward pass
            outputs = model(inp_data)
            # Calculate and record loss and metrics
            metric_step.append(metric(outputs, labels).item())
            loss_step.append(criterion(outputs, labels).item())

    val_loss_epoch = np.mean(loss_step)
    val_metric = np.mean(metric_step)

    return val_loss_epoch, val_metric


def train(model, train_loader, val_loader, criterion, metric, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_metric = -1
    model = model.to(args.device)
    dict_log = {"train_metric": [], "val_metric": [],
                "train_loss": [], "val_loss": []}
    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        train_loss, train_metric = train_one_epoch(
            model, optimizer, train_loader, args.device, criterion, metric)
        val_loss, val_metric,  = validate(
            model, val_loader, criterion, metric, args)

        msg = (f'\nEp {epoch}/{args.epochs}: metric : Train:{train_metric:.3f} \t Val:{val_metric:.2f}\
                || Loss: Train {train_loss:.3f} \t Val {val_loss:.3f}\n')

        if args.verbose:
            print(msg)

        dict_log["train_metric"].append(train_metric)
        dict_log["val_metric"].append(val_metric)
        dict_log["train_loss"].append(train_loss)
        dict_log["val_loss"].append(val_loss)

        if val_metric > best_val_metric and args.save_models:
            best_val_metric = val_metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'metric': val_metric,
            }, f'{args.exp_name}_best_model_min_val_loss.pth')

    return dict_log
