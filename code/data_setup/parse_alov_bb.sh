i=0
while read line
do
    files[ $i ]="$line"        
    (( i++ ))
done < <(find $1 -type f)


echo ${#files[@]}
echo ${files[0]} | cut -d / -f 3 

for file in "${files[@]}"
do 
    sub_dir="$(echo $file | cut -d / -f 3 | cut -d . -f 1),"
    main_dir="$(echo $file | cut -d / -f 2),"
    cat $file | tr " " "," | nl -s ${sub_dir} | cut -c7- | \
        nl -s ${main_dir} | cut -c7- >> $2
done 

