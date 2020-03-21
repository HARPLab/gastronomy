rsync -avtu --exclude="log*" --exclude="cuda_ops*"  --exclude="checkpoints" --exclude="__py**"  ./*  cmu:~/projects/pytorch_disco/
