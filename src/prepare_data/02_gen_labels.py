
import os
import json
from moviepy.editor import VideoFileClip

def prefix(filename):
    """a.jpg -> a"""
    pos = filename.rfind(".")
    if pos == -1:
        return filename
    return filename[:pos]

def basename(filename):
    """a/b/c.jpg -> c"""
    return prefix(os.path.basename(filename))

def remove_suffix(filepath):
    """a/b/c.jpg -> a/b/c"""
    return os.path.join(os.path.dirname(filepath), basename(filepath))

def generate_labels(filenames):
    cur_labels = []
    for video_name in filenames:
        audio_name = remove_suffix(video_name) + ".wav"
        motion_name = remove_suffix(video_name) + ".pkl"
        if os.path.exists(audio_name) and os.path.exists(motion_name) and os.path.exists(video_name):
            item = {
                "video_name": video_name,
                "audio_name": audio_name,
                "motion_name": motion_name,
            }
            cur_labels.append(item)
        else:
            print(motion_name)
            continue
    return cur_labels

def filter_by_duration(video_path, min_duration=8.01):
    try:
        clip = VideoFileClip(video_path)
        return clip.duration >= min_duration
    except:
        return False


total_labels = []
root_dir = "/media/oem/12TB/JoyVASA/data"
filenames = [os.path.join(root_dir, filename) for filename in os.listdir(root_dir) if filename.endswith("mp4") ]
cur_labels = generate_labels(filenames)
total_labels += cur_labels

print(f"{len(total_labels)}")

total_labels = sorted(total_labels, key=lambda x: x["video_name"])
num_train_labels = int(0.8 * len(total_labels))
with open('train.json', 'w', encoding='utf-8') as f:
    json.dump(total_labels[:num_train_labels], f, ensure_ascii=False, indent=4)
with open('test.json', 'w', encoding='utf-8') as f:
    json.dump(total_labels[num_train_labels:], f, ensure_ascii=False, indent=4)