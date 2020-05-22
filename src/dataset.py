import os
import glob
import pandas as pd 
import numpy as np

from mrcnn import model as modellib, utils


class ShipDataset( utils.Dataset ):

    def __init__( self, shape ):

        """ 
        Constructor
        """

        # call base
        super( ShipDataset, self ).__init__()
        self._shape = shape
        return


    def load_info( self, data_path ):

        """ 
        Load a subset of the Kaggle Airbus ship dataset
        data_path: train / test path 
        """

        # add single ship class
        self.add_class("ship", 1, "ship")
        
        image_path = os.path.join( data_path, 'images' )
        annotation_path = os.path.join( data_path, 'annotations' )

        # find annotations
        files = glob.glob( os.path.join( annotation_path, '*.csv' ) )
        for f in files:

            # check image exists
            image_filename = os.path.basename( f ).replace( '.csv', '.jpg' )
            if os.path.exists( os.path.join( image_path, image_filename ) ):

                polygons = []

                # extract object masks from annotation
                df = pd.read_csv( f )
                for idx, row in df.iterrows():
                    polygons.append( row[ 'EncodedPixels'].split() )

            # call base function to add info
            self.add_image(
                "ship",
                image_id=image_filename,  # use file name as a unique image id
                path=image_path,
                width=self._shape[1], height=self._shape[0],
                polygons=polygons)

        return


    def load_mask( self, image_id ):

        """
        Generate instance masks for an image.
        Returns:
        masks: bool array of shape [height, width, instance count] with one mask per instance
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # if not ship dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info[ 'source' ] != 'ship':
            return super(ShipDataset, self).load_mask(image_id)

        # create array of run length encoded 2d masks
        masks = []
        for p in image_info[ 'polygons' ] :
            masks.append( self.decodeRle( p ) )

        # stack mask list into 3d numpy array
        masks = np.dstack( masks )

        return masks.astype(np.bool), np.ones([masks.shape[-1]], dtype=np.int32)


    def image_reference(self, image_id):

        """
        Return the path of the image.
        """

        # return path
        info = self.image_info[image_id]
        if info[ 'source' ] == 'ship':
            return info[ 'path' ]
        else:
            super( ShipDataset, self ).image_reference( image_id )


    def decodeRle( self, rle ):

        """
        rle: run-length as string formated (start length)
        shape: (height,width) of array to return 
        Returns numpy array, 1 - mask, 0 - background
        """

        # ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode

        # parse starts and lengths
        starts, lengths = [np.asarray(x, dtype=int) for x in (rle[0:][::2], rle[1:][::2])]
        starts -= 1
        ends = starts + lengths
        mask = np.zeros( self._shape[0] * self._shape[1], dtype=np.uint8 )

        # set pixels inside boundary to 1
        for lo, hi in zip(starts, ends):
            mask[lo:hi] = 1
        
        # align to RLE direction
        return mask.reshape( self._shape ).T  
