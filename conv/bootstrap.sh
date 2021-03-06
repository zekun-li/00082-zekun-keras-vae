source ~/.bashrc
python2 bootstrap.py 
python2 train_fixval_conv.py --ifxscale \
--iflocalconv \
--bottleneck-filter-size 1 \
--bsize 32 \
--lr 0.0001 \
--train-steps-per-epoch 500 \
--val-steps-per-epoch 500 \
--ith 1 \
--saved-weights 'weights/0_lr0.0001_conv14_nomeanfile_meanloss_xscale_localconv_bfs1x1-best-51-0.03.hdf5'
