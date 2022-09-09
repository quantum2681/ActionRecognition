import torch

def evaluate(model, val_loader, criterion):
    model.eval()
    tot_count = 0
    tot_loss = 0
    tot_correct = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        for i, (data) in enumerate(val_loader):
            input, target = [d.to(device) for d in data]
            target = target.type(torch.LongTensor).to(device)

            output = model(input)
            batch_size = input.size(0)

            print(output, target)

            loss = criterion(output, target)

            preds = output.argmax(dim=1)
            correct = sum(preds == target)

            tot_count += batch_size
            tot_loss += loss.item()
            tot_correct += correct

    evaluation = {
        'loss': tot_loss / tot_count,
        'acc': tot_correct / tot_count
    }

    return evaluation