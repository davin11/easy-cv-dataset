if [ ! -d voc2007_trainval ]
then
  wget -nc http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  mkdir voc2007_trainval
  tar -xvf './VOCtrainval_06-Nov-2007.tar' -C voc2007_trainval | awk 'BEGIN {ORS=" "} {if(NR%200==0)print "."}'
  echo 'done trainval'
fi

if [ ! -d voc2007_test ]
then
  wget -nc http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  mkdir voc2007_test
  tar -xvf './VOCtest_06-Nov-2007.tar' -C voc2007_test | awk 'BEGIN {ORS=" "} {if(NR%200==0)print "."}'
  echo 'done test'
fi
