import glob
import os
import shutil
import argparse
import hashlib
import traceback


def get_image_hash(file):
    BUF_SIZE = 65536
    sha1 = hashlib.sha1()
    with open(file, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)
        image_hash = sha1.hexdigest()
    return image_hash


def main(args):
    image_root = args.image_root_dir

    image_files = glob.glob(os.path.join(image_root, "*"))
    image_hash_map = {}
    for i, image_file in enumerate(image_files):
        if i % 100 == 0:
            print("{}/{}".format(i, len(image_files)), image_file)

        try:
            image_hash = get_image_hash(image_file)
        except:
            traceback.print_exc()
            continue
        if image_hash not in image_hash_map:
            image_hash_map[image_hash] = []
        image_hash_map[image_hash].append(image_file)

    dup_cnt = 0
    dup_cnt_total = 0
    cp_files = 0
    print("중복 검사")
    if args.test and args.test_dir:
        os.makedirs(args.test_dir, exist_ok=True)
    for i, image_hash in enumerate(image_hash_map):
        if len(image_hash_map[image_hash]) > 1:
            dup_cnt += 1
            dup_cnt_total += len(image_hash_map[image_hash])
            print(image_hash_map[image_hash])

            start_idx = 1
            if args.test and args.test_dir:
                start_idx = 0
            for j in range(start_idx, len(image_hash_map[image_hash])):
                if args.test and args.test_dir:
                    shutil.copy(image_hash_map[image_hash][j],
                                os.path.join(args.test_dir, f"{image_hash}_{j}.jpg"))
                else:
                    os.unlink(image_hash_map[image_hash][j])
        cp_files += 1
    print("중복 이미지 종류", dup_cnt)
    print("중복 이미지 파일 갯수", dup_cnt_total)
    print("Unique 이미지 갯수", cp_files)

    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_root_dir', type=str, default='D:\dataset\hair_crawling_naver_images_front_face_thr_0.8_no_mask')
    parser.add_argument('--test', action="store_true", default=False)
    parser.add_argument('--test_dir', type=str,
                        default='D:\dataset\hair_crawling_naver_images_dup_test')

    main(parser.parse_args())
