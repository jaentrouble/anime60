import numpy as np

def frame_to_patch(frame, patch_size, overlap):
    """
    split a frame into patches of patch_size

    Arguments
    ---------
    frame : np.array
        shape : (H,W,C)
    patch_size : tuple of two ints
        format : (H, W)
    overlap : int
    """
    patch_h, patch_w = patch_size
    frame_h, frame_w, frame_c = frame.shape

    patch_num_h = (frame_h) // (patch_h-2*overlap)
    if (frame_h)%(patch_h-2*overlap) > 0:
        patch_num_h += 1
        extra_pad_h = (patch_h-2*overlap)-(frame_h)%(patch_h-2*overlap)
    else:
        extra_pad_h = 0

    patch_num_w = (frame_w) // (patch_w-2*overlap)
    if (frame_w)%(patch_w-2*overlap) > 0:
        patch_num_w += 1
        extra_pad_w = (patch_w-2*overlap)-(frame_w)%(patch_w-2*overlap)
    else:
        extra_pad_w = 0

    padded = np.pad(
        frame,
        ((overlap,extra_pad_h+overlap),(overlap,extra_pad_w+overlap),(0,0)),
        constant_values=0
    )

    patches = np.zeros((patch_num_h,patch_num_w,patch_h,patch_w,frame_c),
                        dtype=frame.dtype)
    for i in range(patch_num_h):
        for j in range(patch_num_w):
            patches[i,j] = padded[
                i*(patch_h-2*overlap):(i+1)*(patch_h-2*overlap)+2*overlap,
                j*(patch_w-2*overlap):(j+1)*(patch_w-2*overlap)+2*overlap,
            ]
    patches = patches.reshape((-1,patch_h,patch_w,frame_c))
    return patches

def frame_to_patch_on_batch(frame, patch_size, overlap):
    """
    split a frame into patches of patch_size on batch

    Arguments
    ---------
    frame : np.array
        shape : (N,H,W,C)
    patch_size : tuple of two ints
        format : (H, W)
    overlap : int

    Return
    ------
    patches : np.array
        shape : (N*num_patches,H,W,C)
    """
    patch_h, patch_w = patch_size
    N, frame_h, frame_w, frame_c = frame.shape

    patch_num_h = (frame_h) // (patch_h-2*overlap)
    if (frame_h)%(patch_h-2*overlap) > 0:
        patch_num_h += 1
        extra_pad_h = (patch_h-2*overlap)-(frame_h)%(patch_h-2*overlap)
    else:
        extra_pad_h = 0

    patch_num_w = (frame_w) // (patch_w-2*overlap)
    if (frame_w)%(patch_w-2*overlap) > 0:
        patch_num_w += 1
        extra_pad_w = (patch_w-2*overlap)-(frame_w)%(patch_w-2*overlap)
    else:
        extra_pad_w = 0

    padded = np.pad(
        frame,
        (
            (0,0),
            (overlap,extra_pad_h+overlap),
            (overlap,extra_pad_w+overlap),
            (0,0)
        ),
        constant_values=0
    )

    patches = np.zeros((N,patch_num_h,patch_num_w,patch_h,patch_w,frame_c),
                        dtype=frame.dtype)
    for i in range(patch_num_h):
        for j in range(patch_num_w):
            patches[:,i,j] = padded[
                :,
                i*(patch_h-2*overlap):(i+1)*(patch_h-2*overlap)+2*overlap,
                j*(patch_w-2*overlap):(j+1)*(patch_w-2*overlap)+2*overlap,
            ]
    patches = patches.reshape((-1,patch_h,patch_w,frame_c))
    return patches


def patch_to_frame(patches, frame_size, overlap):
    """
    stitch patches into a frame

    Arguments
    ---------
    patches : np.array
        shape : (num_patches, patch_H, patch_W, C)
    frame_size : tuple of two ints
        format : (H, W)
    overlap : int

    Return
    ------
    frame : np.array
        shape : (H,W,C)
    """
    patch_num, patch_h, patch_w, patch_c = patches.shape
    frame_h, frame_w = frame_size

    patch_num_h = (frame_h) // (patch_h-2*overlap)
    if (frame_h)%(patch_h-2*overlap) > 0:
        left_over_h = (frame_h)%(patch_h-2*overlap)
        reshape_h = patch_num_h+1
    else:
        left_over_h = 0
        reshape_h = patch_num_h

    patch_num_w = (frame_w) // (patch_w-2*overlap)
    if (frame_w)%(patch_w-2*overlap) > 0:
        left_over_w = (frame_w)%(patch_w-2*overlap)
        reshape_w = patch_num_w + 1
    else:
        left_over_w = 0
        reshape_w = patch_num_w
    
    assert reshape_h * reshape_w == patch_num,\
            'Patch_number does not match. Check frame_size and/or overlap'

    patches = patches.reshape(
        (reshape_h, reshape_w, patch_h, patch_w, patch_c)
    )

    frame = np.zeros((frame_h, frame_w, patch_c),dtype=patches.dtype)
    for i in range(patch_num_h):
        for j in range(patch_num_w):
            frame[
                i*(patch_h-2*overlap):(i+1)*(patch_h-2*overlap),
                j*(patch_w-2*overlap):(j+1)*(patch_w-2*overlap),
            ] = patches[i,j,overlap:-overlap,overlap:-overlap]
    if left_over_h > 0:
        for j in range(patch_num_w):
            frame[
                -left_over_h:,
                j*(patch_w-2*overlap):(j+1)*(patch_w-2*overlap),
            ] = patches[-1,j,overlap:overlap+left_over_h,overlap:-overlap]
    if left_over_w > 0:
        for i in range(patch_num_h):
            frame[
                i*(patch_h-2*overlap):(i+1)*(patch_h-2*overlap),
                -left_over_w:,
            ] = patches[i,-1,overlap:-overlap,overlap:overlap+left_over_w]
    if (left_over_h > 0) and (left_over_w > 0):
        frame[-left_over_h:,-left_over_w:] = patches[
            -1,-1,overlap:overlap+left_over_h,overlap:overlap+left_over_w
        ]

    return frame

def patch_to_frame_on_batch(patches, frame_size, overlap):
    """
    stitch patches into a frame on batch

    Arguments
    ---------
    patches : np.array
        shape : (N*num_patches, patch_H, patch_W, C)
    frame_size : tuple of two ints
        format : (H, W)
    overlap : int

    Return
    ------
    frames : np.array
        shape : (N,H,W,C)
    """
    total_patch_num, patch_h, patch_w, patch_c = patches.shape
    frame_h, frame_w = frame_size

    patch_num_h = (frame_h) // (patch_h-2*overlap)
    if (frame_h)%(patch_h-2*overlap) > 0:
        left_over_h = (frame_h)%(patch_h-2*overlap)
        reshape_h = patch_num_h+1
    else:
        left_over_h = 0
        reshape_h = patch_num_h

    patch_num_w = (frame_w) // (patch_w-2*overlap)
    if (frame_w)%(patch_w-2*overlap) > 0:
        left_over_w = (frame_w)%(patch_w-2*overlap)
        reshape_w = patch_num_w + 1
    else:
        left_over_w = 0
        reshape_w = patch_num_w
    
    assert total_patch_num % (reshape_h * reshape_w)==0,\
            'Patch_number does not match. Check frame_size and/or overlap'

    N = total_patch_num // (reshape_h * reshape_w)
    patches = patches.reshape(
        (N, reshape_h, reshape_w, patch_h, patch_w, patch_c)
    )

    frames = np.zeros((N, frame_h, frame_w, patch_c),dtype=patches.dtype)
    for i in range(patch_num_h):
        for j in range(patch_num_w):
            frames[
                :,
                i*(patch_h-2*overlap):(i+1)*(patch_h-2*overlap),
                j*(patch_w-2*overlap):(j+1)*(patch_w-2*overlap),
            ] = patches[:,i,j,overlap:-overlap,overlap:-overlap]
    if left_over_h > 0:
        for j in range(patch_num_w):
            frames[
                :,
                -left_over_h:,
                j*(patch_w-2*overlap):(j+1)*(patch_w-2*overlap),
            ] = patches[:,-1,j,overlap:overlap+left_over_h,overlap:-overlap]
    if left_over_w > 0:
        for i in range(patch_num_h):
            frames[
                :,
                i*(patch_h-2*overlap):(i+1)*(patch_h-2*overlap),
                -left_over_w:,
            ] = patches[:,i,-1,overlap:-overlap,overlap:overlap+left_over_w]
    if (left_over_h > 0) and (left_over_w > 0):
        frames[:,-left_over_h:,-left_over_w:] = patches[
            :,-1,-1,overlap:overlap+left_over_h,overlap:overlap+left_over_w
        ]

    return frames


if __name__ == '__main__':
    f = np.random.random((720,1280,3))
    s = frame_to_patch(f, (320,320), 60)
    print(s.shape)
    f_prime = patch_to_frame(s, (720,1280), 60)
    print(f_prime.shape)
    print(np.all(f==f_prime))
    fs = np.random.random((10,720,1280,3))
    ss = frame_to_patch_on_batch(fs, (320,320), 60)
    print(ss.shape)
    fs_prime = patch_to_frame_on_batch(ss, (720,1280), 60)
    print(fs_prime.shape)
    print(np.all(fs==fs_prime))