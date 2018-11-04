class Config(object):

      def __init__(self, type_of_model):

        if type_of_model == 'LE-NET':

            # Hyperparameters
            self.lr = 1e-5
            self.l2 = 1e-3
            self.dropout = 0.75
            self.batch_size = 32
            self.mini_batch_size = 32
            self.epochs = 1000
            self.threshold = 0.5

            # Data Processing
            self.image_size = 64
            self.channels = 3
            self.val_size = 0.1
            self.flag_horiz = True

            # Saver
            self.model_name = 'LeNet'
            self.ckpt_path = 'ckpt/' + self.model_name

        elif type_of_model == 'DEEP-CONVNET':

            # Hyperparameters
            self.lr = 1e-5
            self.l2 = 1e-3
            self.dropout = 0.5
            self.batch_size = 128
            self.mini_batch_size = 128
            self.epochs = 1000
            self.threshold = 0.5

            # Data Processing
            self.image_size = 64
            self.channels = 3
            self.val_size = 0.1
            self.flag_horiz = True

            self.model_name = 'Deep-Convnet'
            self.ckpt_path = 'ckpt/' + self.model_name
        elif type_of_model == 'ALEX-NET':

            # Hyperparameters
            self.lr = 0.001
            self.dropout = 0.5
            self.batch_size = 128
            self.mini_batch_size = 128
            self.epochs = 100
            self.threshold = 0.5
            self.train_layers = ['fc8', 'fc7']
            self.weights_path = 'bvlc_alexnet.npy'
            self.MEAN = [104., 117., 124.]

            # Data Processing
            self.image_size = 227
            self.channels = 3
            self.val_size = 0.1
            self.flag_horiz = True

            self.model_name = 'Alex-Net'
            self.ckpt_path = 'ckpt/' + self.model_name
