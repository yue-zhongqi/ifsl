wget https://www.dropbox.com/s/wuxb1wlahado3nq/cifar-fs-splits.zip?dl=0
mv cifar-fs-splits.zip?dl=0 cifar-fs-splits.zip
unzip cifar-fs-splits.zip
rm cifar-fs-splits.zip

python get_cifarfs.py
mv cifar-fs-splits/val1000* cifar-fs/

wget https://www.dropbox.com/s/g9ru5ac5tpupvg6/netFeatBest62.561.pth?dl=0
mv netFeatBest62.561.pth?dl=0 netFeatBest62.561.pth
mkdir ../ckpts
mkdir ../ckpts/CIFAR-FS
mv netFeatBest62.561.pth ../ckpts/CIFAR-FS/
