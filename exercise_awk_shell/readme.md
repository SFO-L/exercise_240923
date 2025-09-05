##### 任务0：将数据集拆分成多行（原始数据集仅有一行），每行包含10个单词（列）。
- 代码：task0.sh
- 结果：text8_split.txt

##### 任务1：输出第三列所有数据（各列用\t分隔）
```
awk -f task1.awk text8_split.txt > outputs/task1_output.txt
```
- 代码：task1.awk
- 结果：task1_output.txt

##### 任务2：输出每10行数据（每10行）
```
awk -f task2.awk text8_split.txt  > outputs/task2_output.txt
```
- 代码：task2.awk
- 结果：task2_output.txt

##### 任务3：若第四列数值超过10000则输出（各列用\t分隔）
```
awk -f task3.awk text8_split.txt > outputs/task3_output.txt
```
- 代码：task3.awk
- 结果：task3_output

##### 任务4：统计第三列的平均单词数（各列用\t分隔，单词用“”分隔）
```
awk -f task4.awk text8_split.txt > outputs/task4_output.txt
```
- 代码：task4.awk
- 结果：task4_output.txt

##### 任务5.1：为给定语料库构建词汇表
- 代码：bash build_vocab.sh
- 结果：vocabulary.txt

##### 任务5.2根据词汇表ID对语料库进行标注
```
awk -f label_corpus.awk vocabulary.txt text8_split.txt > outputs/text8_labeled.txt
```
- 代码：label_corpus.awk
- 结果：text8_labeled.txt

