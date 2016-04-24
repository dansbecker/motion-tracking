#!/bin/bash

read_xml () {
    local IFS=\>
    read -d \< ENTITY CONTENT
}

entities=( folder filename database width height name subcategory xmin xmax ymin ymax )

firstline="folder,filename,database,width,height,name,subcategory,xmin,xmax,ymin,ymax"
echo $firstline >> $2


i=0
while read line
do
    files[ $i ]="$line"        
    (( i++ ))
done < <(find $1 -type f)

echo ${#files[@]}

for file in "${files[@]}"
do 
    while read_xml; 
    do 
        if [[ " ${entities[@]} " =~ " ${ENTITY} " ]]; then   
            eval $ENTITY=$CONTENT
        fi 
        if [[ $ENTITY = "/object" ]]; then 
            out=""
            for entity in "${entities[@]}"
            do 
                if [[ $entity = "subcategory" ]]; then
                    eval sub_var=\$$entity
                    if [ -z $sub_var ]; then
                        sub_var=None
                    fi 
                    out+="$sub_var,"
                else  
                    eval var=\$$entity
                    out+="$var,"
                fi 
            done
            echo ${out:0:${#out}-1} >> $2
        fi 
    done < $file 
done
