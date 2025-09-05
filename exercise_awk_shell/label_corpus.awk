NR == FNR {
   id_map[$2] = $1
   next
}
{  
   line = ""
   for(i=1;i <= NF;i++){
     if($1 in id_map){
        line = line id_map[$1]
     }
     else{
        print "ERROR OOV" > "/dev/stderr"
        line = line "OOV"
     }
     if (i < NF)line = line "\t"
   }
   print line
}
