import json_tricks as json
import numpy as np
from dataset.JointsDataset import JointsDataset
import os

class AnimeDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        self.num_joints = 21
        self.flip_pairs = [[5,9], [6,10], [7,11], [8,12], [13,17], [14,18], [15,19], [16,20]]
        self.aspect_ratio = 1
        self.pixel_std = 200 
        self.image_set = image_set
        self.db = self._get_db()
        
    def _xywh2cs(self,x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale
        
    def _get_db(self):
        file_name = "./data/animepose/"
        json_base_path = "./data/animepose/"
        with open(json_base_path + "train.json") as f:
            train_json = json.load(f)
        with open(json_base_path + "test.json") as f:
            test_json = json.load(f)
        with open(json_base_path + "val.json") as f:
            val_json = json.load(f)

        all_json = train_json + test_json + val_json
        
        gt_db = []
        if self.image_set == "dummy":
            return gt_db
        for j in all_json:
#             print(j)
            image_name = j['file_name'].split("/")[2]
#             print(image_name)
            img_path = self.root+ "images/" + image_name
    
            if image_name in os.listdir(self.root + "images/"):
                new_joints, box = j['points'], (0,0,j['height'],j['width'])
                x, y, w, h = box
                c, s = self._xywh2cs(x, y, w, h)
                if c[0] != -1:
                    c[1] = c[1] + 15 * s[1]
                    s = s * 1.15

                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)
                if self.image_set != 'test':
                    joints = np.array(list(new_joints.values()))
                    joints_vis = [1] * self.num_joints
                    assert len(joints) == self.num_joints, \
                        'joint num diff: {} vs {}'.format(len(joints),
                                                          self.num_joints)

                    joints_3d[:, 0:2] = joints[:, 0:2]
                    joints_3d_vis[:, 0] = joints_vis[:]
                    joints_3d_vis[:, 1] = joints_vis[:]
                gt_db.append({
                    'image': img_path,
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                    })
        return gt_db
    
    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        return {'Null': 0}, 0
    