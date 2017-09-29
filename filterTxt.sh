#/bin/zsh

file=$1

#建立临时文件
tempfile=`mktemp temp.XXXX`
TempFile=`mktemp Temp.XXXX`

#去除英文数字与中英文标点符号
cat $file | sed -E 's/[a-zA-Z0-9[:punct:]]+//g' > $tempfile
#[:punct:]会连中文标点符号也去除，所以使用[!-~]
#去除英文标点
#cat $file | sed -E 's/[!-~]+//g' > $tempfile

#去除重复空格, 行头空格和行尾空格
#cat $tempfile | sed -E 's/^ +//g; s/ +$//g; s/ +/ /g' >$TempFile
#cat $tempfile | sed -E 's/^[[:blank:]]+//g; s/[[:blank:]]+$//g; s/[[:blank:]]+/ /g' > $TempFile
cat $tempfile | sed -E 's/[[:blank:]]+//g' > $TempFile
#去除空白行
cat $TempFile | sed '/^ *$/d' > $tempfile
#去除空（）对, 转换引号
cat $tempfile | sed 's/（ *）//g; s/「/“/g; s/」/”/g' > $TempFile

#将繁体转换成为简体
opencc -i $TempFile -o $tempfile -c t2s.json

#去除一些特殊字符
cat $tempfile | sed 's/[ŋｘɪäö³سМšɛº]//g' > $TempFile

# 打印输出文件
#cat $tempfile
rm $file
cp $TempFile $file

#删除临时文件
rm $tempfile
rm $TempFile
