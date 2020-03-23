rsync -avtu --exclude="log*" --exclude="cuda_ops*"  --exclude="checkpoints" --exclude="__py**"  ./*  aws:/projects/repos/pytorch_disco/
