import os
import configparser

def set_default_parameters(config):
    config.add_section('Input Output')
    config.set('Input Output', 'checkpoint_dir', './results/cheb_VAE_optimal_sigma_VAE')
    config.set('Input Output', 'template', '../project/template/mesh_template5k_edited.obj')
    config.set('Input Output', 'root_dir', '../project/data/mesh_edited')
    config.set('Input Output', 'label_file', '../project/files/files.txt')
    config.set('Input Output', 'error_file', '../project/files/error.txt')
    config.set('Input Output', 'log_file', '/log.txt')
    config.set('Input Output', 'nb_patient', 198)
    
    config.set('Input Output', 'type', 'cheb_VAE')
    config.set('Input Output', 'num_classes', '2')
    config.set('Input Output', 'num_style', '10')
    config.set('Input Output', 'model', 'optimal_sigma_VAE')



    config.add_section('ChebModel  Parameters')

    config.set('ChebModel  Parameters', 'test_size', 0.3)
    config.set('ChebModel  Parameters', 'eval', 'False')
    config.set('ChebModel  Parameters', 'checkpoint_file', '')
    config.set('ChebModel  Parameters', 'n_layers', '2')
    config.set('ChebModel  Parameters', 'num_hidden', '32')

    
    config.set('ChebModel  Parameters', 'downsampling_factors', '4, 4')
    config.set('ChebModel  Parameters', 'polygon_order', '6, 6, 6')
    config.set('ChebModel  Parameters', 'num_conv_filters', '16, 16, 16')
    config.set('ChebModel  Parameters', 'workers_thread', 6)
    config.set('ChebModel  Parameters', 'optimizer', 'adam')
    config.set('ChebModel  Parameters', 'random_seeds', 2020)


    config.add_section('DGCNN Parameters')
    config.set('DGCNN Parameters', 'K', 5)
    config.set('DGCNN Parameters', 'emb_dims', '16,16,16')
    config.set('DGCNN Parameters', 'dropout', 0.5)


    config.add_section('GAT Parameters')
    config.set('GAT Parameters', 'attention_head', 3)


    config.add_section('Learning Parameters')
    config.set('Learning Parameters', 'batch_size', 16)
    config.set('Learning Parameters', 'learning_rate', 1e-3)
    config.set('Learning Parameters', 'learning_rate_decay', 0.99)
    config.set('Learning Parameters', 'weight_decay', 5e-4)
    config.set('Learning Parameters', 'epoch', 300)




def read_config(fname):
    if not os.path.exists(fname):
        print('Config not found %s' % fname)
        return

    config = configparser.RawConfigParser()
    config.read(fname)

    config_parms = {}
    config_parms['root_dir'] = config.get('Input Output', 'root_dir')
    config_parms['checkpoint_dir'] = config.get('Input Output', 'checkpoint_dir')
    config_parms['template'] = config.get('Input Output', 'template')
    config_parms['label_file'] = config.get('Input Output', 'label_file')
    config_parms['error_file'] = config.get('Input Output', 'error_file')
    config_parms['nb_patient'] = config.getint('Input Output', 'nb_patient')
    config_parms['log_file'] = config_parms['checkpoint_dir'] + config.get('Input Output', 'log_file')
    config_parms['type'] = config.get('Input Output', 'type')
    config_parms['num_classes'] = config.getint('Input Output', 'num_classes')
    config_parms['num_style'] = config.getint('Input Output', 'num_style')
    config_parms['model'] = config.get('Input Output', 'model')

    config_parms['test_size'] = config.getfloat('ChebModel  Parameters', 'test_size')
    config_parms['eval'] = config.getboolean('ChebModel  Parameters', 'eval')
    config_parms['checkpoint_file'] = config.get('ChebModel  Parameters', 'checkpoint_file')
    config_parms['n_layers'] = config.getint('ChebModel  Parameters', 'n_layers')
    config_parms['num_hidden'] = config.getint('ChebModel  Parameters', 'num_hidden')
    config_parms['downsampling_factors'] =  [int(x) for x in config.get('ChebModel  Parameters', 'downsampling_factors').split(',')]
    config_parms['num_conv_filters'] = [int(x) for x in config.get('ChebModel  Parameters', 'num_conv_filters').split(',')]
    config_parms['workers_thread'] = config.getint('ChebModel  Parameters', 'workers_thread')
    config_parms['optimizer'] = config.get('ChebModel  Parameters', 'optimizer')
    config_parms['random_seeds'] = config.getint('ChebModel  Parameters', 'random_seeds')
    config_parms['polygon_order'] = [int(x) for x in config.get('ChebModel  Parameters', 'polygon_order').split(',')]


    config_parms['K'] = config.getint('DGCNN Parameters', 'K')
    config_parms['emb_dims'] = [int(x) for x in config.get('DGCNN Parameters', 'emb_dims').split(',')] 
    config_parms['dropout'] = config.getfloat('DGCNN Parameters', 'dropout')

    config_parms['attention_head'] = config.getint('GAT Parameters', 'attention_head')

    config_parms['batch_size'] = config.getint('Learning Parameters', 'batch_size')
    config_parms['learning_rate'] = config.getfloat('Learning Parameters', 'learning_rate')
    config_parms['learning_rate_decay'] = config.getfloat('Learning Parameters', 'learning_rate_decay')
    config_parms['weight_decay'] = config.getfloat('Learning Parameters', 'weight_decay')
    config_parms['epoch'] = config.getint('Learning Parameters', 'epoch')
    return config_parms



if __name__ == '__main__':
 #   pkg_path, _ = os.path.split(os.path.realpath(__file__))
    config_fname = os.path.join('./files', 'default.cfg')

    print('Writing default config file - %s' % config_fname)
    with open(config_fname, 'w') as configfile:
        config = configparser.RawConfigParser()
        set_default_parameters(config)
        config.write(configfile)
        configfile.close()



