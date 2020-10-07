# First-time setup

Install tensorboardX

`pip install tensorboardX`

Install moveipy:

`pip install moviepy`

Install Scikit-image:

`pip install scikit-image`

# Tensorboard

With some cpu resources, on say node `0-16`, run tensorboard with a command like `./tb.sh 3456`.

On the head node, open a tunnel to your tensorboard node, with a command like `./tunnel.sh 3456 0-16`.

# Development

To develop new features and ideas, you will usually run things in `CUSTOM` mode. Run `./custom_go.sh` to do this. If you do not have an `exp_custom.py` file, you should create one. You can copy from any other experiments file. For example, you may want to start with `cp exp_carla_sta.py exp_custom.py`.

Note that `exp_custom.py` is in the `.gitignore` for this repo! This is because custom experiments are private experiments -- and do not usually last long. Once you get a solid result that you would like others to be able to reproduce, add it to one of the main `exp_whatever.py` files and push it to the repo.

<!-- `from spatial_correlation_sampler import SpatialCorrelationSampler`
 -->