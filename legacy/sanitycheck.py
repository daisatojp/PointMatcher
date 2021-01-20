import os
import os.path as osp
import argparse
import json
from glob import glob
from collections import Counter
from PointMatcher.data.matching import Matching


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('annot_dir')
    parser.add_argument('--correction', action='store_true')
    args = parser.parse_args()

    matching = Matching(args.annot_dir)

    num_1 = 0
    keys_1 = []
    group_paths = glob(osp.join(args.annot_dir, 'groups', '*.json'))
    for group_path in group_paths:
        with open(group_path, 'r') as f:
            group = json.load(f)
        num_1 += len(group['keypoints'])
        keys_1 += group['keypoints']

    num_2 = 0
    keys_2 = []
    key_num_total = 0
    view_paths = glob(osp.join(args.annot_dir, 'views', '*.json'))
    for view_path in view_paths:
        with open(view_path, 'r') as f:
            view = json.load(f)
        key_num_total += len(view['keypoints'])
        for keypoint in view['keypoints']:
            if keypoint['group_id'] is not None:
                num_2 += 1
                keys_2 += [[view['id'], keypoint['id']]]

    print('key_num_total', key_num_total)
    print('num_1', num_1)
    print('num_2', num_2)
    print('len(keys_1)', len(keys_1))
    print('len(keys_2)', len(keys_2))
    print('keys_1 duplicates', [item for item, count in Counter(map(tuple, keys_1)).items() if count > 1])
    print('keys_2 duplicates', [item for item, count in Counter(map(tuple, keys_2)).items() if count > 1])

    print('check consistency between view and group file')
    for group_path in group_paths:
        with open(group_path, 'r') as f:
            group = json.load(f)
        gid = group['id']
        for gk in group['keypoints']:
            keypoint = matching.get_keypoint(gk[0], gk[1])
            if keypoint['group_id'] != gid:
                print('wrong vid={}, kid={}, gid={}'.format(gk[0], gk[1], gid))

    print('check duplicate view_id in a group')
    for group_path in group_paths:
        with open(group_path, 'r') as f:
            group = json.load(f)
        view_ids = [keypoint[0] for keypoint in group['keypoints']]
        if 0 < len([item for item, count in Counter(view_ids).items() if count > 1]):
            print('wrong gid={}'.format(group['id']))

    print('check empty group')
    for group_path in group_paths:
        with open(group_path, 'r') as f:
            group = json.load(f)
        if len(group['keypoints']) == 0:
            print('wrong gid={}'.format(group['id']))

    print('check group ids that keypoints of views have')
    list_of_group_id = matching.get_list_of_group_id()
    for view_id in matching.get_list_of_view_id():
        keypoints = matching.get_keypoints(view_id)
        for keypoint in keypoints:
            gid = keypoint['group_id']
            if (gid is not None) and (gid not in list_of_group_id):
                print('wrong vid={}'.format(view_id))
                if args.correction:
                    matching.set_keypoint_group_id(view_id, keypoint['id'], None)
                    print('correct')

    if args.correction:
        matching.save()

    print('sanity check finish.')


if __name__ == '__main__':
    main()
