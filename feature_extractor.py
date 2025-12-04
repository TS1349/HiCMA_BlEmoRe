import os
from collections import namedtuple
import torch
import timm
import utils
import numpy as np

#need this otherwise can't find avit_dim512_patch16_160_a256
import modeling_finetune_av
from kinetics_av import VideoClsDatasetFrame

NUM_CLASSES = 0
NUM_FRAMES = 16
NUM_SEGMENTS = 2
TUBELET_SIZE = 2

INIT_SCALE = 0.1
DEPTH = 10
AUDIO_DEPTH = 10
FUSION_DEPTH = 2

MODEL_PATH="./saved/model/pretraining/voxceleb2/audio_visual/hicmae_pretrain_base/checkpoint-99.pth"


SAMPLING_RATE = 4
TEST_NUM_SEGMENT = 2
TEST_NUM_CROP = 2

INPUT_SIZE = 160
AUDIO_INPUT_SIZE = 256


NUM_MEL_BINS = 128
FREQM = 0
TIMEM = 0

NUM_SAMPLE = 1

device = "cuda:0"

model = timm.create_model(
"avit_dim512_patch16_160_a256",
pretrained=False,
num_classes=NUM_CLASSES,
all_frames= NUM_FRAMES,
tubelet_size=TUBELET_SIZE,
drop_rate=0.0,
drop_path_rate=0.0,
attn_drop_rate=0.0,
drop_block_rate=None,
init_scale=INIT_SCALE,
attn_type="joint",
depth=DEPTH,
depth_audio=AUDIO_DEPTH,
fusion_depth=FUSION_DEPTH,
use_mean_pooling=True,
)


checkpoint = torch.load(MODEL_PATH, map_location="cpu")
checkpoint_model = checkpoint["model"]

state_dict = model.state_dict()
metadata = getattr(state_dict, '_metadata', None)
utils.load_state_dict(model, checkpoint_model)
model.to(device)


print("================")
norm_stats = {'audioset': [-4.2677393, 4.5689974], 'k400': [-4.2677393, 4.5689974],
              'esc50': [-6.6268077, 5.358466], 'speechcommands': [-6.845978, 5.5654526]}
audio_conf_train = {'num_mel_bins': NUM_MEL_BINS,
                    'target_length': AUDIO_INPUT_SIZE,
                    'freqm': FREQM,
                    'timem': TIMEM,
                    'mean': norm_stats['audioset'][0],
                    'std': norm_stats['audioset'][1],
                    'noise': False,
                    }
audio_conf_val = {'num_mel_bins': NUM_MEL_BINS,
                  'target_length': AUDIO_INPUT_SIZE,
                  'freqm': 0,
                  'timem': 0,
                  'mean': norm_stats['audioset'][0],
                  'std': norm_stats['audioset'][1],
                  'noise': False,
                  }
audio_conf = {
    'train': audio_conf_train, 'validation': audio_conf_val, 'test': audio_conf_val
}

args = namedtuple("args",
                  ["data_set",
                   "reprob",
                   "audio_conf",
                   "roll_mag_aug",
                   "num_sample",
                   "aa",
                   "train_interpolation",
                   "remode",
                   "recount",
                   ])(
    data_set = "blemore",
    reprob = 0.25,
    audio_conf = audio_conf,
    roll_mag_aug = True,
    num_sample = NUM_SAMPLE,
    aa = "rand-m7-n4-mstd0.5-inc1",
    train_interpolation="bicubic",
    remode = "pixel",
    recount = 1,
)


test_mode = True

DATA_PATH = "./saved/data/blemore/audio_visual/split06/"

def extract_features(dataset,out_dir):
    model.eval()
    with torch.inference_mode():
        for i in range(0,len(dataset.dataset_samples)):
            video, audio, sample  = dataset.get_for_feature_extraction(i)
            sample = sample.split("/")[-1]
            out_name = f"{out_dir}{sample}"
            if os.path.exists(f"{out_name}.npy"):
                print(f"skipping {sample} since it already exists")
                continue
            video = video.to(device)
            audio = audio.to(device)
            assert video.shape[0] == audio.shape[0]
            output = model(video, audio)
            output = output.to("cpu").numpy()

            np.save(out_name, output)
            print(f"{sample}.npy saved [shape = {output.shape}]")

feature_dir = "./feature_dirs/"
data_dir = feature_dir + "data/"
os.makedirs(feature_dir, exist_ok = True)
os.makedirs(data_dir, exist_ok = True)

dataset_train = VideoClsDatasetFrame(
    anno_path=f"{DATA_PATH}train.csv" ,
    data_path='/',
    mode="test",
    clip_len=NUM_FRAMES,
    frame_sample_rate=SAMPLING_RATE,
    num_segment=1,
    test_num_segment=TEST_NUM_SEGMENT,
    test_num_crop=TEST_NUM_CROP,
    num_crop=1 if not test_mode else 3,
    keep_aspect_ratio=True,
    crop_size=INPUT_SIZE,
    short_side_size=INPUT_SIZE,
    new_height=256, # me: actually no use
    new_width=320, # me: actually no use
    args=args,
    task='regression',
    file_ext="bmp"
)

extract_features(dataset_train, data_dir)

test_dir = feature_dir + "test/"
os.makedirs(test_dir, exist_ok = True)



dataset_test = VideoClsDatasetFrame(
    anno_path=f"{DATA_PATH}test.csv" ,
    data_path='/',
    mode="test",
    clip_len=NUM_FRAMES,
    frame_sample_rate=SAMPLING_RATE,
    num_segment=1,
    test_num_segment=TEST_NUM_SEGMENT,
    test_num_crop=TEST_NUM_CROP,
    num_crop=1 if not test_mode else 3,
    keep_aspect_ratio=True,
    crop_size=INPUT_SIZE,
    short_side_size=INPUT_SIZE,
    new_height=256, # me: actually no use
    new_width=320, # me: actually no use
    args=args,
    task='regression',
    file_ext="bmp"
)

extract_features(dataset_test, test_dir)
