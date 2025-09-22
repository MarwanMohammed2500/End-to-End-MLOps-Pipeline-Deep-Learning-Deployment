from metrics import set_metrics
from tqdm.auto import tqdm
import torch

def train_test_loop(epochs, model, loss_fn, optimizer, train_dataloader, test_dataloader, writer=None, device="cpu", verbose=False):
    accuracy_score, f1_score, precision, recall = set_metrics(num_classes=9, task="multiclass", device=device)
    for epoch in tqdm(range(epochs)):
        print(f"Epoch #{epoch}")
        train_loss = 0
        test_loss = 0
        train_acc = 0
        test_acc = 0

        for batch, (X_train, y_train) in enumerate(train_dataloader):
            model.train()
            X_train, y_train = X_train.to(device), y_train.squeeze().to(device)
            y_pred = model(X_train).squeeze()
            loss = loss_fn(y_pred, y_train)
            train_loss += loss.item()
            accuracy_score.update(torch.softmax(y_pred, dim=1).argmax(dim=1), y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose and batch > 0 and batch%400==0:
                print(f"[INFO] Batch #{batch}: Average Train Loss: {(train_loss/(batch+1)):.3f}")
        train_acc = accuracy_score.compute()
        accuracy_score.reset()

        model.eval()
        with torch.inference_mode():
            for X_test, y_test in test_dataloader:
                X_test, y_test = X_test.to(device), y_test.to(device)

                test_pred = model(X_test).squeeze()
                loss = loss_fn(test_pred, y_test.squeeze())
                test_loss += loss.item()
                accuracy_score.update(torch.softmax(test_pred, dim=1).argmax(dim=1), y_test)
        

        average_train_loss = train_loss/len(train_dataloader)

        average_test_loss = test_loss/len(test_dataloader)
        test_acc = accuracy_score.compute()
        accuracy_score.reset()

        print(f"[INFO] Average Train Loss: {average_train_loss:.3f}, Train Accuracy: {train_acc:.2%}, Average Test Loss: {average_test_loss:.3f}, Test Accuracy: {test_acc:.2%}")

        if writer is not None:
            writer.add_scalars(
                "Average Train & Test Loss",
                {"Train Loss": average_train_loss, "Test Loss": average_test_loss},
                epoch+1
                            )
            
            writer.add_scalars(
                "Average Train & Test Accuracy",
                {"Train Accuracy": train_acc, "Test Accuracy": test_acc},
                epoch+1
                            )
    return model
