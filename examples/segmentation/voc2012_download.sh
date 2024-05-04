if [ ! -d voc2012_trainval ]
then
  wget -nc http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  mkdir voc2012_trainval
  tar -xvf './VOCtrainval_11-May-2012.tar' -C voc2012_trainval
fi

