import os
import random
import argparse
import matplotlib.pyplot as plt

from mrcnn import utils
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn.visualize import display_images

from config import ShipConfig
from dataset import ShipDataset


class Predict:

    def __init__( self, args ):

        """
        constructor
        """

        # initialise config
        self._config = ShipConfig()

        # initialise model with configuration - write model and logs to out path
        self._model = modellib.MaskRCNN(    mode="inference", 
                                            config=self._config, 
                                            model_dir=args.out_path )

        return


    def process( self, args ):

        """
        Run inference with trained model
        """

        # load validation / test dataset
        test_ds = ShipDataset( (768,768) )
        test_ds.load_info( os.path.join( args.data_path, 'test' ) )
        test_ds.prepare()

        # Load weights
        print("Loading weights ", args.model_pathname)
        self._model.load_weights(args.model_pathname, by_name=True)

        for sample in range ( 10 ):

            # pick random image
            image_id = random.choice( test_ds.image_ids )
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(  test_ds, 
                                                                                        self._config, 
                                                                                        image_id, 
                                                                                        use_mini_mask=False )
            # get image info
            info = test_ds.image_info[ image_id ]
            print("image ID: {}.{} ({}) {}".format( info["source"], 
                                                    info["id"], 
                                                    image_id, 
                                                    test_ds.image_reference(image_id) ) )

            # run object detection
            results = self._model.detect([image], verbose=1)
            r = results[0]
    
            # compute AP over range 0.5 to 0.95 and print it
            utils.compute_ap_range( gt_bbox, 
                                    gt_class_id, 
                                    gt_mask,
                                    r['rois'], 
                                    r['class_ids'], 
                                    r['scores'], 
                                    r['masks'],
                                    verbose=1)
    
            # display results
            visualize.display_instances(    image, 
                                            r['rois'], 
                                            r['masks'], 
                                            r['class_ids'], 
                                            test_ds.class_names, 
                                            r['scores'], 
                                            title="Predictions",
                                            show_bbox=False)

            # display actual vs predicted differences
            visualize.display_differences(  image, 
                                            gt_bbox, 
                                            gt_class_id, 
                                            gt_mask,
                                            r['rois'], 
                                            r['class_ids'], 
                                            r['scores'], 
                                            r['masks'], 
                                            test_ds.class_names, 
                                            title="Actual vs Predict Difference" )

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

    return parser.parse_args(args)


def main():

    """
    main path of execution
    """

    # parse arguments
    args = parseArguments()
    obj = Predict( args )

    # execute training
    obj.process( args )

    return


# execute main
if __name__ == '__main__':
    main()

