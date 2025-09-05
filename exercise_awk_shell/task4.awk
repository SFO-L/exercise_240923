
# 任务4：计算第3列中单词的平均数量
{
    n = split($3, arr, " ")
    total += n
    count++
}
END {
    if (count > 0) {
        print "第3列平均单词数:", total / count
    }
}
