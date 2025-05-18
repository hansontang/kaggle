import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
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
    df['Phrase'] = df['Phrase'].fillna('').astype(str)
    return df


train = clean_phrase_data(train)
test = clean_phrase_data(test)

# 检查类别分布
print("情感类别分布：")
print(train['Sentiment'].value_counts(normalize=True))


# 填充序列到固定长度
def Pad_sequences(phrase_to_int, seq_length):
    pad_sequences = np.zeros((len(phrase_to_int), seq_length), dtype=int)
    for idx, row in tqdm(enumerate(phrase_to_int), total=len(phrase_to_int)):
        pad_sequences[idx, :len(row)] = np.array(row)[:seq_length]
    return pad_sequences


# 提取语料库并将短语转换为整数序列
def Corpus_Extr(df):
    print('构建语料库...')
    corpus = []
    for i in tqdm(range(len(df))):
        phrase = df.Phrase.iloc[i]  # 使用 iloc 按位置索引
        if isinstance(phrase, str):
            corpus.append(phrase.lower().split())
        else:
            corpus.append([])
    corpus = Counter(np.hstack(corpus))
    corpus2 = sorted(corpus, key=corpus.get, reverse=True)
    print('将语料库转换为整数...')
    vocab_to_int = {word: idx for idx, word in enumerate(corpus2, 1)}
    print('将短语转换为整数...')
    phrase_to_int = []
    for i in tqdm(range(len(df))):
        phrase = df.Phrase.iloc[i]  # 使用 iloc 按位置索引
        if isinstance(phrase, str):
            phrase_to_int.append([vocab_to_int[word] for word in phrase.lower().split()])
        else:
            phrase_to_int.append([])
    return corpus, vocab_to_int, phrase_to_int


# 转换验证集短语，使用训练集的词汇表
def convert_phrases_to_int(df, vocab_to_int):
    print('将验证集短语转换为整数...')
    phrase_to_int = []
    for i in tqdm(range(len(df))):
        phrase = df.Phrase.iloc[i]  # 使用 iloc 按位置索引
        if isinstance(phrase, str):
            phrase_to_int.append([vocab_to_int.get(word, 0) for word in phrase.lower().split()])
        else:
            phrase_to_int.append([])
    return phrase_to_int


# 划分训练集和验证集
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42, stratify=train['Sentiment'])

# 处理训练数据
corpus, vocab_to_int, phrase_to_int = Corpus_Extr(train_df)
train_pad = Pad_sequences(phrase_to_int, 30)

# 处理验证数据，使用训练集的词汇表
val_phrase_to_int = convert_phrases_to_int(val_df, vocab_to_int)
val_pad = Pad_sequences(val_phrase_to_int, 30)


# 数据集类：用于训练和验证数据
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


# 情感RNN模型（优化版）
class SentimentRNN(nn.Module):
    def __init__(self, corpus_size, output_size, embedd_dim, hidden_dim, n_layers):
        super().__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(corpus_size, embedd_dim)
        self.lstm = nn.LSTM(embedd_dim, hidden_dim, n_layers, dropout=0.3, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim * 2, output_size)  # 双向LSTM，隐藏维度翻倍

    def forward(self, x, hidden):
        batch_size = x.size(0)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim * 2)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = out.view(batch_size, -1)
        out = out[:, -5:]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_().to(device))  # 双向LSTM，层数翻倍
        return hidden


# 初始化模型并移动到GPU
vocab_size = len(vocab_to_int) + 1  # 为未知词保留索引0
output_size = 5
embedding_dim = 300  # 适配预训练嵌入
hidden_dim = 256
n_layers = 2
net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
net = net.to(device)
net.train()

# 训练参数
clip = 5
epochs = 50  # 减少epoch，结合早停
batch_size = 64  # 增大批次大小
lr = 0.001  # 降低学习率

# 使用标准交叉熵损失
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# 训练和验证循环
best_val_loss = float('inf')
patience = 10
counter = 0
early_stop = False

for e in range(epochs):
    if early_stop:
        print("早停触发，停止训练")
        break

    # 训练
    train_set = PhraseDataset(train_df, train_pad)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    net.train()
    running_loss = 0.0
    running_acc = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"训练 轮次 {e + 1}/{epochs}"):
        # 动态初始化隐藏状态，匹配实际批次大小
        h = net.init_hidden(inputs.size(0))
        h = tuple([each.data for each in h])
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        output, h = net(inputs, h)
        loss = criterion(output, labels)
        loss.backward()
        running_loss += loss.item()
        running_acc += (output.argmax(dim=1) == labels).float().mean().item()
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

    train_loss = running_loss / len(train_loader)
    train_acc = running_acc / len(train_loader)

    # 验证
    val_set = PhraseDataset(val_df, val_pad)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    net.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"验证 轮次 {e + 1}/{epochs}"):
            # 动态初始化隐藏状态
            h = net.init_hidden(inputs.size(0))
            inputs = inputs.to(device)
            labels = labels.to(device)
            output, h = net(inputs, h)
            loss = criterion(output, labels)
            val_loss += loss.item()
            val_acc += (output.argmax(dim=1) == labels).float().mean().item()

    val_loss = val_loss / len(val_loader)
    val_acc = val_acc / len(val_loader)

    print(f"轮次: {e + 1}/{epochs}, 训练损失: {train_loss:.6f}, 训练准确率: {train_acc:.6f}, "
          f"验证损失: {val_loss:.6f}, 验证准确率: {val_acc:.6f}")

    # 学习率调度
    scheduler.step(val_loss)

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(net.state_dict(), 'best_model.pth')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            early_stop = True

# 加载最佳模型
net.load_state_dict(torch.load('best_model.pth'))


# ------------------- 预测test.tsv情感标签 -------------------

# 预处理测试数据
def preprocess_test_data(test_df, vocab_to_int, seq_length=30):
    print('正在处理测试语料库...')
    phrase_to_int_test = []
    for i in tqdm(range(len(test_df))):
        phrase = test_df.Phrase.iloc[i]  # 使用 iloc 按位置索引
        if isinstance(phrase, str):
            phrase_to_int_test.append([
                vocab_to_int.get(word, 0) for word in phrase.lower().split()
            ])
        else:
            phrase_to_int_test.append([])
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
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in tqdm(test_loader, desc="正在预测"):
            # 动态初始化隐藏状态
            h = model.init_hidden(inputs.size(0))
            inputs = inputs.to(device)
            output, h = model(inputs, h)
            predicted_labels = output.argmax(dim=1).cpu().numpy()
            predictions.extend(predicted_labels)
    return predictions


# 主预测流程
print("开始处理测试数据并预测情感标签...")
pad_sequences_test = preprocess_test_data(test, vocab_to_int, seq_length=30)
test_set = TestPhraseDataset(pad_sequences_test)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
predicted_sentiments = predict_sentiment(net, test_loader, device)

# 创建提交DataFrame
submission = pd.DataFrame({
    'PhraseId': test['PhraseId'],
    'Sentiment': predicted_sentiments
})

# 保存预测结果到CSV
submission.to_csv('submission.csv', index=False)
print("预测结果已保存至'submission.csv'")