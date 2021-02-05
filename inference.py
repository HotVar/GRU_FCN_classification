import pandas as pd
import torch.nn

def inference():
    data = pd.read_csv('open/test_features.csv')
    submission = pd.read_csv('open/sample_submission.csv').set_index('id')

    device = 'cpu'
    unique_ids = data['id'].unique()
    softmax = torch.nn.Softmax(dim=1)

    for i, id_ in enumerate(unique_ids):
        print(f'({i+1}/{len(unique_ids)})')
        X = data[data['id'] == id_].iloc[:, 2:].values
        X = torch.Tensor(X).to(device).unsqueeze(0)

        model = torch.load('2021-02-05_200.pth', map_location=torch.device(device))
        model.to(device)
        model.eval()
        model.GRU.batch_size = 1
        model.GRU.device = device
        pred = softmax(model(X))
        pred = pred.squeeze().detach().numpy()
        submission.loc[id_, :] = pred

    submission.reset_index().to_csv('submission.csv', index=False)

if __name__=='__main__':
    inference()