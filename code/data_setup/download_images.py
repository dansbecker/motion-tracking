import concurrent.futures
import itertools
import json
import pandas as pd
import socket
import urllib.request
import os
from os.path import join, exists

class ImageDownloader(object):

    def __init__(self, image_url_df, target_path='../../data/images', nb_workers=32):
        self._failed_to_capture_path = '../../work/failed_to_capture_images.json'
        self.image_urls = image_url_df.itertuples()
        self.target_path = target_path
        self.nb_workers = nb_workers
        self.imgs_to_request = image_url_df.shape[0]
        self.imgs_previously_captured = 0
        self.imgs_requested = 0
        self.failed_img_requests = 0
        socket.setdefaulttimeout(5)

    def __enter__(self):
        try:
            if not exists(self.target_path):
                os.makedirs(self.target_path)
            if not exists('work'):
                os.makedirs('work')
            with open(self._failed_to_capture_path, 'r') as f:
                self.failed_to_capture = json.load(f)
        except:
            self.failed_to_capture = []
        return(self)

    def __exit__(self, *args):
        with open(self._failed_to_capture_path, 'w') as f:
            json.dump(self.failed_to_capture, f)

    def run(self):
        print('Images to request: ', self.imgs_to_request)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.nb_workers) as worker_pool:
            for _, fname, img_src in self.image_urls:
                if ((self.imgs_requested + self.imgs_previously_captured) % 5000 == 0):
                    print('Images requested this run: ', self.imgs_requested)
                    print('Images skipped because already captured: ', self.imgs_previously_captured)
                    print('Failed image requests: ', self.failed_img_requests)
                worker_pool.submit(self.get_one_img(fname, img_src))

    def get_one_img(self, fname, url):
        local_img_path = join(self.target_path, fname)
        if exists(local_img_path):
            self.imgs_previously_captured += 1
            return
        try:
            self.imgs_requested += 1
            url, _ = urllib.request.urlretrieve(url, local_img_path)
        except:
            self.failed_img_requests += 1
            self.failed_to_capture.append((url, local_img_path))


if __name__ == "__main__":
    xml_files_by_dir = (i[2] for i in os.walk('../../data/ILSVRC2014_DET_bbox_train'))
    bbox_xml_files = itertools.chain(*xml_files_by_dir)
    imgs_with_bbox_xml = set([f.replace('.xml', '') for f in bbox_xml_files])
    # We had access to ilsvrc12_urls.txt but it is a subset of fall11_ruls.txt
    url_list = pd.read_table('../../data/fall11_urls.txt',
                             names=['image_fname', 'url'],
                             error_bad_lines=False,
                             warn_bad_lines=True,
                             encoding='ISO-8859-1') # just a guess. utf-8 causes error
    imgs_to_download = url_list.query("image_fname in @imgs_with_bbox_xml")
    with ImageDownloader(imgs_to_download) as img_downloader:
        img_downloader.run()
