if [ ! -d voc2007_trainval ]
then
  wget -nc http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  mkdir voc2007_trainval
  tar -xf './VOCtrainval_06-Nov-2007.tar' -C voc2007_trainval --checkpoint=.200
  echo 'done trainval'
fi

if [ ! -d voc2007_test ]
then
  wget -nc http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  mkdir voc2007_test
  tar -xf './VOCtest_06-Nov-2007.tar' -C voc2007_test --checkpoint=.200
  echo 'done test'
fi
