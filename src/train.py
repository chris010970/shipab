import os
import time
import random
import argparse

import mrcnn.model as modellib

from config import ShipConfig
from dataset import ShipDataset

class Train:

    def __init__( self, args ):

        """
        constructor
        """

        # initialise config
        self._config = ShipConfig()

        # initialise model with configuration - write model and logs to out path
        self._model = modellib.MaskRCNN(    mode="training", 
                                            config=self._config, 
                                            model_dir=args.out_path )
        return


    def process( self, args ):

        """
        setup and initialise mask-rcnn objects and commence training
        """
        
        # load training dataset.
        train_ds = ShipDataset( (768,768) )
        train_ds.load_info( os.path.join( args.data_path, 'train' ) )
        train_ds.prepare()

        # load validation / test dataset
        test_ds = ShipDataset( (768,768) )
        test_ds.load_info( os.path.join( args.data_path, 'test' ) )
        test_ds.prepare()

        # load weights
        if args.model_pathname == 'last':

            # attempt to load last set of weights and continue training
            self._model.load_weights( self._model.find_last(), by_name=True )
        
        else:

            if os.path.basename( args.model_pathname ) == 'mask_rcnn_coco.h5':

                # load original COCO weights - skip incompatible layers due to different number of classes
                self._model.load_weights(   args.model_pathname,
                                            by_name=True,
                                            exclude=[
                                                "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask",
                                                "rpn_model"  # because anchor's ratio has been changed
                                            ] )
            else:

                # load weights direct from pathname
                self._model.load_weights( args.model_pathname, by_name=True )

        # commence training - top layer only
        print("Training network heads")
        self._model.train(  train_ds, 
                            test_ds,
                            learning_rate=self._config.LEARNING_RATE,
                            epochs=args.epochs,
                            layers='heads' )

        # fine tune all layers
        print("Training all layers")
        self._model.train(  train_ds, 
                            test_ds,
                            learning_rate=self._config.LEARNING_RATE / 10,
                            epochs=2,
                            layers='all' )

        # save final weights
        if args.model_pathname is not None:
            pathname = args.model_pathname.replace( '.h5', '-{}-{}.h5'.format ( args.epochs, time.strftime("%Y%m%d-%H%M%S") ) )
            self._model.keras_model.save_weights( pathname )
        
        return


def parseArguments(args=None):

    """
    parse command line arguments
    """

    # parse configuration
    parser = argparse.ArgumentParser(description='preparation')
    parser.add_argument('data_path', action="store")
    parser.add_argument('model_pathname', action="store")
    parser.add_argument('out_path', action="store")

    # optional settings
    parser.add_argument('--epochs', action="store", default=100 )

    return parser.parse_args(args)


def main():

    """
    main path of execution
    """

    # parse arguments
    args = parseArguments()
    obj = Train( args )

    # execute training
    obj.process( args )

    return


# execute main
if __name__ == '__main__':
    main()

