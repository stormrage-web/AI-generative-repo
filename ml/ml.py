import sys

from urllib.parse import urlparse
from torch.hub import download_url_to_file, get_dir
import os
import time
import cv2
import math

from PIL import Image

from rembg import remove

import matplotlib.pyplot as plt
import keras_ocr
import cv2
import math
import numpy as np

# For inpainting

import numpy as np
import multiprocessing
from transparent_background import Remover

import torch

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
)

remover = Remover()

def remove_background_func(img):
    return remove(img)

def download_model(url=LAMA_MODEL_URL):
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(model_dir):
        os.makedirs(os.path.join(model_dir, "hub", "checkpoints"))
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)
    return cached_file


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def numpy_to_bytes(image_numpy: np.ndarray) -> bytes:
    data = cv2.imencode(".jpg", image_numpy)[1]
    image_bytes = data.tobytes()
    return image_bytes


def load_img(img_bytes, gray: bool = False):
    nparr = np.frombuffer(img_bytes, np.uint8)
    if gray:
        np_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    else:
        np_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if len(np_img.shape) == 3 and np_img.shape[2] == 4:
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGRA2RGB)
        else:
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

    return np_img


def norm_img(np_img):
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img


def resize_max_size(
    np_img, size_limit: int, interpolation=cv2.INTER_CUBIC
) -> np.ndarray:
    # Resize image's longer size to size_limit if longer size larger than size_limit
    h, w = np_img.shape[:2]
    if max(h, w) > size_limit:
        ratio = size_limit / max(h, w)
        new_w = int(w * ratio + 0.5)
        new_h = int(h * ratio + 0.5)
        return cv2.resize(np_img, dsize=(new_w, new_h), interpolation=interpolation)
    else:
        return np_img


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    )



try:
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
except:
    pass

NUM_THREADS = str(multiprocessing.cpu_count())

os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS
if os.environ.get("CACHE_DIR"):
    os.environ["TORCH_HOME"] = os.environ["CACHE_DIR"]

#BUILD_DIR = os.environ.get("LAMA_CLEANER_BUILD_DIR", "./lama_cleaner/app/build")

# For Seam-carving

from scipy import ndimage as ndi

SEAM_COLOR = np.array([255, 200, 200])    # seam visualization color (BGR)
SHOULD_DOWNSIZE = True                    # if True, downsize image for faster carving
DOWNSIZE_WIDTH = 500                      # resized image width if SHOULD_DOWNSIZE is True
ENERGY_MASK_CONST = 100000.0              # large energy value for protective masking
MASK_THRESHOLD = 10                       # minimum pixel intensity for binary mask
USE_FORWARD_ENERGY = True                 # if True, use forward energy algorithm

device = torch.device("cpu")
model_path = "big-lama.pt"
model = torch.jit.load(model_path, map_location="cpu")
model = model.to(device)
model.eval()


########################################
# UTILITY CODE
########################################


def visualize(im, boolmask=None, rotate=False):
    vis = im.astype(np.uint8)
    if boolmask is not None:
        vis[np.where(boolmask == False)] = SEAM_COLOR
    if rotate:
        vis = rotate_image(vis, False)
    cv2.imshow("visualization", vis)
    cv2.waitKey(1)
    return vis

def resize(image, width):
    dim = None
    h, w = image.shape[:2]
    dim = (width, int(h * width / float(w)))
    image = image.astype('float32')
    return cv2.resize(image, dim)

def rotate_image(image, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(image, k)


########################################
# ENERGY FUNCTIONS
########################################

def backward_energy(im):
    """
    Simple gradient magnitude energy map.
    """
    xgrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=1, mode='wrap')
    ygrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=0, mode='wrap')

    grad_mag = np.sqrt(np.sum(xgrad**2, axis=2) + np.sum(ygrad**2, axis=2))

    # vis = visualize(grad_mag)
    # cv2.imwrite("backward_energy_demo.jpg", vis)

    return grad_mag

def forward_energy(im):
    """
    Forward energy algorithm as described in "Improved Seam Carving for Video Retargeting"
    by Rubinstein, Shamir, Avidan.
    Vectorized code adapted from
    https://github.com/axu2/improved-seam-carving.
    """
    h, w = im.shape[:2]
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))

    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)

    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU

    for i in range(1, h):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)

        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)

    # vis = visualize(energy)
    # cv2.imwrite("forward_energy_demo.jpg", vis)

    return energy

########################################
# SEAM HELPER FUNCTIONS
########################################

def add_seam(im, seam_idx):
    """
    Add a vertical seam to a 3-channel color image at the indices provided
    by averaging the pixels values to the left and right of the seam.
    Code adapted from https://github.com/vivianhylee/seam-carving.
    """
    h, w = im.shape[:2]
    output = np.zeros((h, w + 1, 3))
    for row in range(h):
        col = seam_idx[row]
        for ch in range(3):
            if col == 0:
                p = np.mean(im[row, col: col + 2, ch])
                output[row, col, ch] = im[row, col, ch]
                output[row, col + 1, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]
            else:
                p = np.mean(im[row, col - 1: col + 1, ch])
                output[row, : col, ch] = im[row, : col, ch]
                output[row, col, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]

    return output

def add_seam_grayscale(im, seam_idx):
    """
    Add a vertical seam to a grayscale image at the indices provided
    by averaging the pixels values to the left and right of the seam.
    """
    h, w = im.shape[:2]
    output = np.zeros((h, w + 1))
    for row in range(h):
        col = seam_idx[row]
        if col == 0:
            p = np.mean(im[row, col: col + 2])
            output[row, col] = im[row, col]
            output[row, col + 1] = p
            output[row, col + 1:] = im[row, col:]
        else:
            p = np.mean(im[row, col - 1: col + 1])
            output[row, : col] = im[row, : col]
            output[row, col] = p
            output[row, col + 1:] = im[row, col:]

    return output

def remove_seam(im, boolmask):
    h, w = im.shape[:2]
    boolmask3c = np.stack([boolmask] * 3, axis=2)
    return im[boolmask3c].reshape((h, w - 1, 3))

def remove_seam_grayscale(im, boolmask):
    h, w = im.shape[:2]
    return im[boolmask].reshape((h, w - 1))

def get_minimum_seam(im, mask=None, remove_mask=None):
    """
    DP algorithm for finding the seam of minimum energy. Code adapted from
    https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
    """
    h, w = im.shape[:2]
    energyfn = forward_energy if USE_FORWARD_ENERGY else backward_energy
    M = energyfn(im)

    if mask is not None:
        M[np.where(mask > MASK_THRESHOLD)] = ENERGY_MASK_CONST

    # give removal mask priority over protective mask by using larger negative value
    if remove_mask is not None:
        M[np.where(remove_mask > MASK_THRESHOLD)] = -ENERGY_MASK_CONST * 100

    seam_idx, boolmask = compute_shortest_path(M, im, h, w)

    return np.array(seam_idx), boolmask

def compute_shortest_path(M, im, h, w):
    backtrack = np.zeros_like(M, dtype=np.int_)


    # populate DP matrix
    for i in range(1, h):
        for j in range(0, w):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    # backtrack to find path
    seam_idx = []
    boolmask = np.ones((h, w), dtype=np.bool_)
    j = np.argmin(M[-1])
    for i in range(h-1, -1, -1):
        boolmask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_idx.reverse()
    return seam_idx, boolmask

########################################
# MAIN ALGORITHM
########################################

def seams_removal(im, num_remove, mask=None, vis=False, rot=False):
    for _ in range(num_remove):
        seam_idx, boolmask = get_minimum_seam(im, mask)
        if vis:
            visualize(im, boolmask, rotate=rot)
        im = remove_seam(im, boolmask)
        if mask is not None:
            mask = remove_seam_grayscale(mask, boolmask)
    return im, mask


def seams_insertion(im, num_add, mask=None, vis=False, rot=False):
    seams_record = []
    temp_im = im.copy()
    temp_mask = mask.copy() if mask is not None else None

    for _ in range(num_add):
        seam_idx, boolmask = get_minimum_seam(temp_im, temp_mask)
        if vis:
            visualize(temp_im, boolmask, rotate=rot)

        seams_record.append(seam_idx)
        temp_im = remove_seam(temp_im, boolmask)
        if temp_mask is not None:
            temp_mask = remove_seam_grayscale(temp_mask, boolmask)

    seams_record.reverse()

    for _ in range(num_add):
        seam = seams_record.pop()
        im = add_seam(im, seam)
        if vis:
            visualize(im, rotate=rot)
        if mask is not None:
            mask = add_seam_grayscale(mask, seam)

        # update the remaining seam indices
        for remaining_seam in seams_record:
            remaining_seam[np.where(remaining_seam >= seam)] += 2

    return im, mask

########################################
# MAIN DRIVER FUNCTIONS
########################################

def seam_carve(im, dy, dx, mask=None, vis=False):
    im = im.astype(np.float64)
    h, w = im.shape[:2]
    assert h + dy > 0 and w + dx > 0 and dy <= h and dx <= w

    if mask is not None:
        mask = mask.astype(np.float64)

    output = im

    if dx < 0:
        output, mask = seams_removal(output, -dx, mask, vis)

    elif dx > 0:
        output, mask = seams_insertion(output, dx, mask, vis)

    if dy < 0:
        output = rotate_image(output, True)
        if mask is not None:
            mask = rotate_image(mask, True)
        output, mask = seams_removal(output, -dy, mask, vis, rot=True)
        output = rotate_image(output, False)

    elif dy > 0:
        output = rotate_image(output, True)
        if mask is not None:
            mask = rotate_image(mask, True)
        output, mask = seams_insertion(output, dy, mask, vis, rot=True)
        output = rotate_image(output, False)

    return output


def object_removal(im, rmask, mask=None, vis=False, horizontal_removal=False):
    im = im.astype(np.float64)
    rmask = rmask.astype(np.float64)
    if mask is not None:
        mask = mask.astype(np.float64)
    output = im

    h, w = im.shape[:2]

    if horizontal_removal:
        output = rotate_image(output, True)
        rmask = rotate_image(rmask, True)
        if mask is not None:
            mask = rotate_image(mask, True)

    while len(np.where(rmask > MASK_THRESHOLD)[0]) > 0:
        seam_idx, boolmask = get_minimum_seam(output, mask, rmask)
        if vis:
            visualize(output, boolmask, rotate=horizontal_removal)
        output = remove_seam(output, boolmask)
        rmask = remove_seam_grayscale(rmask, boolmask)
        if mask is not None:
            mask = remove_seam_grayscale(mask, boolmask)

    num_add = (h if horizontal_removal else w) - output.shape[1]
    output, mask = seams_insertion(output, num_add, mask, vis, rot=horizontal_removal)
    if horizontal_removal:
        output = rotate_image(output, False)

    return output



def s_image(im,mask,vs,hs,mode="resize"):
    im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
    mask = 255-mask[:,:,3]
    h, w = im.shape[:2]
    if SHOULD_DOWNSIZE and w > DOWNSIZE_WIDTH:
        im = resize(im, width=DOWNSIZE_WIDTH)
        if mask is not None:
            mask = resize(mask, width=DOWNSIZE_WIDTH)

    # image resize mode
    if mode=="resize":
        dy = hs#reverse
        dx = vs#reverse
        assert dy is not None and dx is not None
        output = seam_carve(im, dy, dx, mask, False)


    # object removal mode
    elif mode=="remove":
        assert mask is not None
        output = object_removal(im, mask, None, False, True)

    return output


##### Inpainting helper code

def run(image, mask):
    """
    image: [C, H, W]
    mask: [1, H, W]
    return: BGR IMAGE
    """
    origin_height, origin_width = image.shape[1:]
    image = pad_img_to_modulo(image, mod=8)
    mask = pad_img_to_modulo(mask, mod=8)

    mask = (mask > 0) * 1
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask).unsqueeze(0).to(device)

    start = time.time()
    with torch.no_grad():
        inpainted_image = model(image, mask)

    print(f"process time: {(time.time() - start)*1000}ms")
    cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
    cur_res = cur_res[0:origin_height, 0:origin_width, :]
    cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
    cur_res = cv2.cvtColor(cur_res, cv2.COLOR_BGR2RGB)
    return cur_res


def process_inpaint(image, mask):
    original_shape = image.shape
    interpolation = cv2.INTER_CUBIC

    size_limit = max(image.shape)

    print(f"Origin image shape: {original_shape}")
    image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)
    print(f"Resized image shape: {image.shape}")
    image = norm_img(image)

    mask = mask[:, :, 3]
    mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)
    mask = norm_img(mask)

    res_np_img = run(image, mask)

    return cv2.cvtColor(res_np_img, cv2.COLOR_BGR2RGB)


#The start point will be the mid-point between the top-left corner and
#the bottom-left corner of the box.
#the end point will be the mid-point between the top-right corner and the bottom-right corner.
#The following function does exactly that.
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

def inpainted_text_mask(img, pipeline):
    # read the image
    # img = keras_ocr.tools.read(img_path)

    # Recogize text (and corresponding regions)
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    prediction_groups = pipeline.recognize([img])

    #Define the mask for inpainting
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)

        #For the line thickness, we will calculate the length of the line between
        #the top-left corner and the bottom-left corner.
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))

        #Define the line and inpaint
        cv2.rectangle(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)

    return mask

pipeline = keras_ocr.pipeline.Pipeline()

def get_foreground_as_black_oil(source_img, transparent_background):
  replace_main_mask = np.where(transparent_background[:, :, 3] == 0)
  main_as_black_oil = np.zeros(source_img.shape)
  main_as_black_oil[replace_main_mask] = source_img[replace_main_mask]
  return main_as_black_oil


def REMOVE_INFOGRAPHICS(a_img):
    remove_background = remove_background_func(a_img)
    zeros = np.zeros(remove_background.shape[:2])
    img_text_removed = inpainted_text_mask(
        get_foreground_as_black_oil(a_img, remove_background).astype('float32'),
        pipeline
    )
    return process_inpaint(
            a_img,
            np.dstack((zeros,) * 3 + (img_text_removed,))
        )

def REMOVE_BACKGR(a_img):
    return remover.process(a_img)


