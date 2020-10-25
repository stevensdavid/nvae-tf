import hashlib
import io
import os
import pickle
import re
import uuid
import requests
from typing import Any
import tensorflow as tf
import math
import html
import glob


def tile_images(images):
    n_images = tf.cast(tf.shape(images)[0], float)
    # Convert to side of square
    n = int(tf.math.floor(tf.math.sqrt(n_images)))
    _, height, width, channels = tf.shape(images)
    images = tf.reshape(images, [n, n, height, width, channels])
    images = tf.transpose(images, perm=[2, 0, 3, 1, 4])
    return tf.reshape(images, [n * height, n * width, channels])


def calculate_log_p(z, mu, sigma):
    normalized_z = (z - mu) / sigma
    log_p = -0.5 * normalized_z * normalized_z - 0.5 * tf.math.log(2*tf.constant(math.pi)) - tf.math.log(sigma)
    return log_p


def softclamp5(x):
    return 5.0 * tf.math.tanh(x / 5.0)  # differentiable clamp [-5, 5]

# Remainder of this file is NVIDIA source code from https://github.com/NVlabs/stylegan2:
# 
# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html
def open_file_or_url(file_or_url):
    if is_url(file_or_url):
        return open_url(file_or_url, cache_dir='.stylegan2-cache')
    return open(file_or_url, 'rb')

def load_pkl(file_or_url):
    with open_file_or_url(file_or_url) as file:
        return pickle.load(file, encoding='latin1')

def is_url(obj: Any, allow_file_urls: bool = False) -> bool:
    """Determine whether the given object is a valid URL string."""
    if not isinstance(obj, str) or not "://" in obj:
        return False
    if allow_file_urls and obj.startswith('file:///'):
        return True
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True


def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert is_url(url, allow_file_urls=True)
    assert num_attempts >= 1

    # Handle file URLs.
    if url.startswith('file:///'):
        return open(url[len('file:///'):], "rb")

    # Lookup from cache.
    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache_dir is not None:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            return open(cache_files[0], "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache_dir is not None:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file) # atomic

    # Return data as file object.
    return io.BytesIO(url_data)