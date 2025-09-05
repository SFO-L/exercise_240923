awk '
BEGIN {
    RS = " "        # 输入记录分隔符为空格
    field_count = 0
}
{
    if (NF == 0) next  # 跳过空字段
    field_count++
    printf "%s", $0   # 输出单词
    if (field_count % 10 == 0) {
        print ""      # 每10个单词换行
    } else {
        printf "\t"   # 否则用制表符分隔
    }
}
' text8 > text8_split.txt

echo "完成task0 生成text8_split.txt"