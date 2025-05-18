import numpy as np
import pandas as pd
import torch
import random
import pickle
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import gc

# 设置设备：优先使用GPU（CUDA），否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用的设备: {device}")

# 加载训练和测试数据
train = pd.read_csv('train.tsv', sep='\t')
test = pd.read_csv('test.tsv', sep='\t')


# 数据清洗：处理Phrase列中的NaN和非字符串值
def clean_phrase_data(df):
    print("清洗数据中的Phrase列...")
    # 将NaN或非字符串值替换为空字符串
    df['Phrase'] = df['Phrase'].fillna('').astype(str)
    return df


train = clean_phrase_data(train)
test = clean_phrase_data(test)


# 提取语料库并将短语转换为整数序列
def Corpus_Extr(df):
    print('构建语料库...')
    corpus = []
    for i in tqdm(range(len(df))):
        phrase = df.Phrase[i]
        # 确保phrase是字符串并处理
        if isinstance(phrase, str):
            corpus.append(phrase.lower().split())
        else:
            corpus.append([])  # 非字符串（如空值）转为空列表
    corpus = Counter(np.hstack(corpus))
    corpus2 = sorted(corpus, key=corpus.get, reverse=True)
    print('将语料库转换为整数...')
    vocab_to_int = {word: idx for idx, word in enumerate(corpus2, 1)}
    print('将短语转换为整数...')
    phrase_to_int = []
    for i in tqdm(range(len(df))):
        phrase = df.Phrase.values[i]
        if isinstance(phrase, str):
            phrase_to_int.append([vocab_to_int[word] for word in phrase.lower().split()])
        else:
            phrase_to_int.append([])  # 非字符串转为空序列
    return corpus, vocab_to_int, phrase_to_int


corpus, vocab_to_int, phrase_to_int = Corpus_Extr(train)


# 填充序列到固定长度
def Pad_sequences(phrase_to_int, seq_length):
    pad_sequences = np.zeros((len(phrase_to_int), seq_length), dtype=int)
    for idx, row in tqdm(enumerate(phrase_to_int), total=len(phrase_to_int)):
        pad_sequences[idx, :len(row)] = np.array(row)[:seq_length]
    return pad_sequences


pad_sequences = Pad_sequences(phrase_to_int, 30)


# 数据集类：用于训练数据
class PhraseDataset(Dataset):
    def __init__(self, df, pad_sequences):
        super().__init__()
        self.df = df
        self.pad_sequences = pad_sequences

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if 'Sentiment' in self.df.columns:
            label = self.df['Sentiment'].values[idx]
            item = self.pad_sequences[idx]
            return item, label
        else:
            item = self.pad_sequences[idx]
            return item


# 情感RNN模型
class SentimentRNN(nn.Module):
    def __init__(self, corpus_size, output_size, embedd_dim, hidden_dim, n_layers):
        super().__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(corpus_size, embedd_dim)
        self.lstm = nn.LSTM(embedd_dim, hidden_dim, n_layers, dropout=0.5, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.act = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.act(out)
        out = out.view(batch_size, -1)
        out = out[:, -5:]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        # 初始化隐藏状态并移动到指定设备
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden


# 初始化模型并移动到GPU
vocab_size = len(vocab_to_int) + 1  # 加1为未知词保留索引0
output_size = 5
embedding_dim = 400
hidden_dim = 256
n_layers = 2
net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
net = net.to(device)  # 将模型移动到GPU
net.train()

# 训练参数
clip = 5
epochs = 200
counter = 0
print_every = 100
lr = 0.01
batch_size = 32


# 自定义损失函数
def criterion(input, target, size_average=True):
    """分类交叉熵，输入为logits，目标为one-hot编码"""
    l = -(target * torch.log(F.softmax(input, dim=1) + 1e-10)).sum(1)
    if size_average:
        l = l.mean()
    else:
        l = l.sum()
    return l


optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# 训练循环
losses = []
accs = []
for e in range(epochs):
    a = np.random.choice(len(train) - 1, 1000)
    train_set = PhraseDataset(train.loc[train.index.isin(np.sort(a))], pad_sequences[a])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 初始化隐藏状态
    h = net.init_hidden(batch_size)
    running_loss = 0.0
    running_acc = 0.0

    # 批次循环
    for idx, (inputs, labels) in enumerate(train_loader):
        counter += 1
        gc.collect()

        # 将隐藏状态移动到GPU
        h = tuple([each.data for each in h])

        # 清除累积梯度
        optimizer.zero_grad()

        # 将输入和标签移动到GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        if inputs.shape[0] != batch_size:
            break

        # 前向传播
        output, h = net(inputs, h)
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=5).float()

        # 计算损失并反向传播
        loss = criterion(output, labels_one_hot)
        loss.backward()
        running_loss += loss.cpu().detach().numpy()
        running_acc += (output.argmax(dim=1) == labels_one_hot.argmax(dim=1)).float().mean()

        # 梯度裁剪，防止梯度爆炸
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        if idx % 20 == 0:
            print("轮次: {}/{}...".format(e + 1, epochs),
                  "步骤: {}...".format(counter),
                  "损失: {:.6f}...".format((running_loss / (idx + 1))))
            losses.append(float(running_loss / (idx + 1)))
            print(f'准确率: {running_acc / (idx + 1)}')
            accs.append(running_acc / (idx + 1))


# ------------------- 预测test.tsv情感标签 -------------------

# 预处理测试数据
def preprocess_test_data(test_df, vocab_to_int, seq_length=30):
    print('正在处理测试语料库...')
    # 将测试短语转换为整数序列
    phrase_to_int_test = []
    for i in tqdm(range(len(test_df))):
        phrase = test_df.Phrase.values[i]
        # 确保phrase是字符串
        if isinstance(phrase, str):
            phrase_to_int_test.append([
                vocab_to_int.get(word, 0) for word in phrase.lower().split()
            ])
        else:
            phrase_to_int_test.append([])  # 非字符串转为空序列
    # 填充序列
    pad_sequences_test = Pad_sequences(phrase_to_int_test, seq_length)
    return pad_sequences_test


# 测试数据集类
class TestPhraseDataset(Dataset):
    def __init__(self, pad_sequences):
        super().__init__()
        self.pad_sequences = pad_sequences

    def __len__(self):
        return len(self.pad_sequences)

    def __getitem__(self, idx):
        return self.pad_sequences[idx]


# 预测情感标签
def predict_sentiment(model, test_loader, device):
    model.eval()  # 设置模型为评估模式
    predictions = []

    with torch.no_grad():  # 禁用梯度计算以进行推理
        for inputs in tqdm(test_loader, desc="正在预测"):
            # 将输入数据移动到GPU
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # 初始化隐藏状态
            h = model.init_hidden(batch_size)

            # 获取模型输出
            output, h = model(inputs, h)

            # 将输出logits转换为预测标签
            predicted_labels = output.argmax(dim=1).cpu().numpy()
            predictions.extend(predicted_labels)

    return predictions


# 主预测流程
print("开始处理测试数据并预测情感标签...")
# 预处理测试数据
pad_sequences_test = preprocess_test_data(test, vocab_to_int, seq_length=30)

# 创建测试数据集和数据加载器
test_set = TestPhraseDataset(pad_sequences_test)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 预测情感标签
predicted_sentiments = predict_sentiment(net, test_loader, device)

# 创建提交DataFrame
submission = pd.DataFrame({
    'PhraseId': test['PhraseId'],
    'Sentiment': predicted_sentiments
})

# 保存预测结果到CSV
submission.to_csv('submission.csv', index=False)
print("预测结果已保存至'submission.csv'")

