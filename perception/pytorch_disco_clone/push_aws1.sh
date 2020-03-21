rsync -avtu --exclude="log*" --exclude="cuda_ops*"  --exclude="checkpoints" --exclude="__py**"  ./*  aws1:/projects/repos/pytorch_disco/
