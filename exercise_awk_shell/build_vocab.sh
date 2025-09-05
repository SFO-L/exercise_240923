cat text8_split.txt | tr '\t' '\n'|sort |uniq -c|sort -nr| \
awk '{
   word = $2
   freq = $1
   print NR-1,word,freq
}' > vocabulary.txt

echo "完成Task5.1 生成vocabulary.txt 有$(wc -l < vocabulary.txt)个词"
