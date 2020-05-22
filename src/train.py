import os

from config import ShipConfig
from dataset import ShipDataset
from mrcnn import model as modellib, utils

class Train:

    def __init__( self, args ):

        """
        constructor
        """

        # initialise config
        self._config = ShipConfig()
        return


    def process( self, args ):

        """
        Train the model.
        """
        
        # load training dataset.
        train_ds = ShipDataset( (768,768) )
        train_ds.load_info( os.path.join( args.data_path, 'train' ) )
        train_ds.prepare()

        # load validation / test dataset
        test_ds = ShipDataset( (768,768) )
        test_ds.load_info( os.path.join( args.data_path, 'test' ) )
        test_ds.prepare()

        # initialise model with configuration - write model and logs to out path
        model = modellib.MaskRCNN(  mode="training", 
                                    config=self._config, 
                                    model_dir=args.out_path )

        if args.model_pathname == 'last':

            # load last trained model and continue training
            model.load_weights( model.find_last(), by_name=True )

        else:

            # load weights trained on MS COCO - skip incompatible layers due to different number of classes
            model.load_weights( args.model_pathname,
                                by_name=True,
                                exclude=[
                                    "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask",
                                    "rpn_model"  # because anchor's ratio has been changed
                                ] )

        # commence training - top layer only
        print("Training network heads")
        model.train(    train_ds, 
                        test_ds,
                        learning_rate=self._config.LEARNING_RATE,
                        epochs=args.epochs,
                        layers='heads' )

        # fine tune all layers
        print("Training all layers")
        model.train(    train_ds, 
                        test_ds,
                        learning_rate=self._config.LEARNING_RATE / 10,
                        epochs=10,
                        layers='all' )

        # save final weights
        model.keras_model.save_weights( args.model_pathname.replace( '.h5', '-final.h5' ) )

        return
