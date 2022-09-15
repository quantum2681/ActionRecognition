import torch

def evaluate(model, val_loader, criterion):
    model.eval()
    tot_count = 0
    tot_loss = 0
    tot_correct = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    max_o = -1e3
    min_o = 1e3
    with torch.no_grad():
        for i, (data) in enumerate(val_loader):
            input = [d.to(device) for d in data[0]]
            target = data[1].type(torch.LongTensor).to(device)

            output = model(input)
            batch_size = input[0].size(0)




            loss = criterion(output, target)

            preds = output.argmax(dim=1)
            correct = sum(preds == target)

            # print(output)
            # print(preds, target, sep='\t')

            if torch.max(output) > max_o:
                max_o = torch.max(output)
            if torch.min(output) < min_o:
                min_o = torch.min(output)

            tot_count += batch_size
            tot_loss += loss.item()
            tot_correct += correct

    print(f'max {max_o:.3f}, min {min_o:3f}, ', end='\t')

    evaluation = {
        'loss': tot_loss / tot_count,
        'acc': tot_correct / tot_count
    }

    return evaluation