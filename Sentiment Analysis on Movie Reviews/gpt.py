import pandas as pd
import random
from openai import OpenAI
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# 初始化 OpenAI 客户端
client = OpenAI(api_key="")

# 读取数据
train_df = pd.read_csv("train.tsv", sep="\t", encoding='utf-8')
test_df = pd.read_csv("test.tsv", sep="\t", encoding='utf-8')
submission_df = pd.read_csv("submission.csv", encoding='utf-8')

# 只选择测试集前100条
test_df = test_df.head(100)

# 合并测试集和真实标签
test_df = test_df.merge(submission_df, on='PhraseId', how='left')

# 只选择训练集中100条样本
few_shot_samples = train_df.sample(n=100, random_state=42)

# 构建 few-shot prompts
few_shot_messages = [
    {
        "role": "system",
        "content": (
            "You are a movie review sentiment classifier.\n"
            "Classify the following reviews based on their sentiment. The sentiment labels are:\n"
            "0-negative, 1-somewhat negative, 2-neutral, 3-somewhat positive, 4-positive.\n"
            "Return only one number representing the sentiment label."
        )
    }
]

# 添加 few-shot 样本到对话中（只用 Phrase 和 Sentiment）
for _, row in few_shot_samples.iterrows():
    few_shot_messages.append({"role": "user", "content": row['Phrase']})
    few_shot_messages.append({"role": "assistant", "content": str(row['Sentiment'])})

# 定义 GPT 调用函数
def get_response(example):
    messages = few_shot_messages + [{"role": "user", "content": example}]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.75,
            max_tokens=5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error processing example: {example}. Error: {e}")
        return "2"  # 返回默认值（中性）以防止代码中断

# 对测试集每条评论进行情感分类，添加进度条
tqdm.pandas()
test_df['gpt_prediction'] = test_df['Phrase'].progress_apply(get_response)

# 过滤掉 Sentiment 缺失的行
test_df = test_df.dropna(subset=['Sentiment'])

# 去除重复短语
test_df = test_df.drop_duplicates(subset=['Phrase'])

# 打印前几条结果
print("测试集预测结果（前5条）：")
print(test_df[['Phrase', 'Sentiment', 'gpt_prediction']].head())

# 确保预测值和真实标签为相同类型（字符串）
test_df['gpt_prediction'] = test_df['gpt_prediction'].astype(str)
test_df['Sentiment'] = test_df['Sentiment'].astype(str)

# 计算准确率
accuracy = accuracy_score(test_df['Sentiment'], test_df['gpt_prediction'])
print(f"\nAccuracy: {accuracy:.4f}")

# 输出分类性能报告
print("\nClassification Report:")
print(classification_report(test_df['Sentiment'], test_df['gpt_prediction'], zero_division=0))
