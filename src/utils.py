import io
import PIL.Image
import numpy as np
import ipywidgets as widgets

def image_to_bytes(a, fmt='png'):
    """
    Convert PNG array to bytes array
    """
    a = np.uint8(a)
    f = io.BytesIO()
    ima = PIL.Image.fromarray(a).save(f, fmt)
    return f.getvalue()


class Displayer():
    """Create a display in Jupyter Notebook
    to observe training.
    """

    def __init__(self):

        self.image_layout = widgets.Layout(display='flex',
                                      flex_flow='row',
                                      align_items='center',
                                      border='solid',
                                      width='25%',
                                      justify_content="center")

        self.image = widgets.Image(layout=self.image_layout)
