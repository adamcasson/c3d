# Easy C3D for Keras
C3D for Keras 2.0 (only Tensorflow backend at the moment) with easy preprocessing and automatic downloading of TF format sports1M weights

## Requirements
* Python 2 or 3
* Keras 2.0+
* skvideo
  * ffmpeg
* scipy
* numpy

## Examples

### Classify videos

```python
import skvideo.io
from c3d import C3D
from sports1M_utils import preprocess_input, decode_predictions

model = C3D(weights='sports1M')

vid_path = 'homerun.mp4'
vid = skvideo.io.vread(vid_path)
# Select 16 frames from video
vid = vid[40:56]
x = preprocess_input(vid)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))
#Predicted: [('baseball', 0.91488838)]
```

### Extract features from videos

```python
import skvideo.io
from c3d import C3D
from keras.models import Model
from sports1M_utils import preprocess_input, decode_predictions

base_model = C3D(weights='sports1M')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc6').output)

vid_path = ‘homerun.mp4'
vid = skvideo.io.vread(vid_path)
# Select 16 frames from video
vid = vid[40:56]
x = preprocess_input(vid)

features = model.predict(x)
```

## References
* [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767)

## Acknowledgements

Thanks to [albertomontesg](https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2) for C3D Sports1M theano weights and Keras code. Thanks to [titu1994](https://github.com/titu1994/Keras-Classification-Models/blob/master/weight_conversion_theano.py) for Theano to Tensorflow weight conversion code.
