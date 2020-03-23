import os
os.environ["exp_name"] = "trainer_occ_lr3"
os.environ["MODE"] = "CLEVR_STA"
os.environ["run_name"] = "ad"
from model_clevr_sta import CLEVR_STA
from model_carla_sta import CARLA_STA
from model_carla_flo import CARLA_FLO
from model_carla_obj import CARLA_OBJ
import hyperparams as hyp
import cProfile
import logging
import ipdb 
st = ipdb.set_trace
logger = logging.Logger('catch_all')

def main():
    
    checkpoint_dir_ = os.path.join("checkpoints", hyp.name)
    st()
    if hyp.do_clevr_sta:
        log_dir_ = os.path.join("logs_clevr_sta", hyp.name)    
    elif hyp.do_carla_sta:
        log_dir_ = os.path.join("logs_carla_sta", hyp.name)
    elif hyp.do_carla_flo:
        log_dir_ = os.path.join("logs_carla_flo", hyp.name)
    elif hyp.do_carla_obj:
        log_dir_ = os.path.join("logs_carla_obj", hyp.name)
    else:
        assert(False) # what mode is this?
    # st()

    if not os.path.exists(checkpoint_dir_):
        os.makedirs(checkpoint_dir_)
    if not os.path.exists(log_dir_):
        os.makedirs(log_dir_)

    try:
        if hyp.do_clevr_sta:
            model = CLEVR_STA(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()        
        elif hyp.do_carla_sta:
            model = CARLA_STA(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_flo:
            model = CARLA_FLO(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_obj:
            model = CARLA_OBJ(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        else:
            assert(False) # what mode is this?

    except (Exception, KeyboardInterrupt) as ex:
        logger.error(ex, exc_info=True)
        log_cleanup(log_dir_)

def log_cleanup(log_dir_):
    log_dirs = []
    for set_name in hyp.set_names:
        log_dirs.append(log_dir_ + '/' + set_name)

    for log_dir in log_dirs:
        for r, d, f in os.walk(log_dir):
            for file_dir in f:
                file_dir = os.path.join(log_dir, file_dir)
                file_size = os.stat(file_dir).st_size
                if file_size == 0:
                    os.remove(file_dir)

if __name__ == '__main__':
    main()
    # cProfile.run('main()')

