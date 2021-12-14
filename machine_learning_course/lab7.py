import torch
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS

def collate_batch():
    pass

class KlasyfikatorTekstu(nn,Module):
    def __init__(self, vocab_size, embed_dim):
        super(KlasyfikatorTekstu, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)

        self.fc = nn.Linear(embed_dim)

        self.optimizer =
        self.scheduler =

model = KlasyfikatorTekstu(len(vocab), 5)


def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    for idx, (label, text, offset) in enumerate(dataloader):
        model.optimizer.zero_grad()
        pred_label = model(text, offset)
        loss = model.loss_function(pred_label, label)
        loss.backward()
        model.optimizer.step()

        total_acc += (pred_label.argmax(1) == label).sum().item()
        total_count += label.size(0)

        if idx % 500 == 0:
            print('Accuracy for', idx, 'in', len(dataloader), ':', total_acc/total_count)
            total_acc, total_count = 0, 0


def evaluate(dataloader):
    model.eval()
    with torch.no_grad():
        total_acc, total_count = 0, 0
        for idx, (label, text, offset) in enumerate(dataloader):
            pred_label = model(text, offset)
            total_acc += (pred_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

    return total_acc/total_count


from torchtext.data.functional import to_map_style_dataset

BATCH_SIZE = 64
EPOCHS = 10

train_iter, test_iter = AG_NEWS()
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

for epoch in range(1, EPOCHS+1):
    train(train_dataloader)
    accu_val = evaluate(test_dataloader)
    model.scheduler.step()
    print('-'*20)
    print('Accuracy for epoch', epoch, ':', accu_val)
    print('-'*20)

ag_news_label = {1: 'World',
                 2: 'Sports',
                 3: 'Business',
                 4: 'Sci/Tec'}

def predict(text):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item()+1

my_fav_news = ''

model = model.to('cpu')
print('This news is', ag_news_label[predict(my_fav_news)])
