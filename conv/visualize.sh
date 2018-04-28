source ~/.bashrc
#python2 bootstrap.py 
python2 visualize_fixval_conv.py --ifxscale \
--bsize 32 \
--train-steps-per-epoch 500 \
--val-steps-per-epoch 500 \
--saved-weights 'weights/0_lr0.0001_conv14_nomeanfile_meanloss_xscale_conv_bfs3x3-best-55-0.02.hdf5'
