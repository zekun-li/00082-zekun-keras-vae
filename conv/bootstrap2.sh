source ~/.bashrc
python2 bootstrap.py 
python2 train_fixval_conv.py --ifxscale \
--bottleneck-filter-size 5 \
--bsize 32 \
--lr 0.0001 \
--train-steps-per-epoch 500 \
--val-steps-per-epoch 500 \
--ith 1 \
--saved-weights 'weights/0_lr0.0001_conv14_nomeanfile_meanloss_xscale_conv_bfs5x5-best-93-0.02.hdf5'
