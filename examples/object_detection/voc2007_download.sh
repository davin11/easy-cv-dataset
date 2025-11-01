if [ ! -d voc2007_trainval ]
then
  wget -nc -O './VOCtrainval_06-Nov-2007.tar' "https://huggingface.co/datasets/davin11/VOC2007/resolve/main/VOCtrainval_06-Nov-2007.tar?download=true"
  mkdir voc2007_trainval
  tar -xf './VOCtrainval_06-Nov-2007.tar' -C voc2007_trainval --checkpoint=.200
  echo 'done trainval'
fi

if [ ! -d voc2007_test ]
then
  wget -nc -O './VOCtest_06-Nov-2007.tar' "https://huggingface.co/datasets/davin11/VOC2007/resolve/main/VOCtest_06-Nov-2007.tar?download=true"
  mkdir voc2007_test
  tar -xf './VOCtest_06-Nov-2007.tar' -C voc2007_test --checkpoint=.200
  echo 'done test'
fi
