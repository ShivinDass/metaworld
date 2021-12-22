from mujoco_py.modder import TextureModder
import cv2

class TextureModderExtended(TextureModder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def set_given_texture(self, name, inp_texture):
        '''
        Args:
        - name (str): name of the texture to replace
        - inp_texture (numpy array): The texture that is going to be used instead
        '''
        bitmap = self.get_texture(name).bitmap
        h, w = bitmap.shape[:2]
        
        # inp_texture = cv2.resize(inp_texture, (w,h))

        '''
        Resizing it this way so that resolution is better otherwise the background is too distorted.
        Hardcoded the values for DAVIS17 dataset and 'behindGripper' camera angle
        '''
        inp_texture = cv2.resize(inp_texture, (24, 13))
        inp_texture = cv2.copyMakeBorder(inp_texture, 167, 12, 4, 4, cv2.BORDER_CONSTANT, value=(0,0,0))
        
        bitmap[:] = inp_texture[:,:,:3]

        self.upload_texture(name)
        return bitmap