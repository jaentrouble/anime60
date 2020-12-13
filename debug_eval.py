import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from model_trainer_half import create_train_dataset, AnimeModel
import flow_models as fm


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

vid_dir = Path('data/cut')
vid_paths = [str(vid_dir/vn) for vn in os.listdir(vid_dir)]
random.shuffle(vid_paths)

model_f = fm.ehrb0_143_32 

frame_size = (320,320)

val_ds = create_train_dataset(vid_paths, frame_size, 24, val_data=True, parallel=6)

inputs= keras.Input((frame_size[1],frame_size[0],6))
dummy_model = AnimeModel(inputs,model_f,[0.5])
loss = keras.losses.MeanAbsoluteError()
dummy_model.compile(
    optimizer='adam',
    loss=loss
)
dummy_model.evaluate(
    val_ds,
    steps=1000
)