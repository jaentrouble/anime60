import tensorflow as tf

@tf.function
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
        shape : (num_patches*N,H,W,C)
    """
    patch_h, patch_w = patch_size
    N= tf.shape(frame)[0]
    frame_h= tf.shape(frame)[1]
    frame_w= tf.shape(frame)[2]
    frame_c= tf.shape(frame)[3]

    need_extra_pad_h = tf.cast((frame_h)%(patch_h-2*overlap) > 0, tf.int32)

    patch_num_h = need_extra_pad_h + (frame_h) // (patch_h-2*overlap)
    extra_pad_h = need_extra_pad_h * \
        ((patch_h-2*overlap)-(frame_h)%(patch_h-2*overlap))

    need_extra_pad_w = tf.cast((frame_w)%(patch_w-2*overlap) > 0, tf.int32)

    patch_num_w = need_extra_pad_w + (frame_w) // (patch_w-2*overlap)
    extra_pad_w = need_extra_pad_w * \
        ((patch_w-2*overlap)-(frame_w)%(patch_w-2*overlap))

    padded = tf.pad(
        frame,
        (
            (0,0),
            (overlap,extra_pad_h+overlap),
            (overlap,extra_pad_w+overlap),
            (0,0)
        ),
        constant_values=0
    )
    
    patches_list = tf.TensorArray(
        padded.dtype,
        size=0,
        dynamic_size=True,
    )
    for i in range(patch_num_h):
        for j in range(patch_num_w):
            patches_list = patches_list.write(patches_list.size(),padded[
                :,
                i*(patch_h-2*overlap):(i+1)*(patch_h-2*overlap)+2*overlap,
                j*(patch_w-2*overlap):(j+1)*(patch_w-2*overlap)+2*overlap,
            ])
    patches = patches_list.concat()
    return patches

@tf.function
def patch_to_frame_on_batch(patches, patch_size, frame_size, overlap):
    """
    stitch patches into a frame on batch

    Arguments
    ---------
    patches : tf.Tensor
        shape : (num_patches*N, patch_H, patch_W, C)
    patch_size : tuple of two ints
        format : (H, W)
    frame_size : tuple of two ints
        format : (H, W)
    overlap : int

    Return
    ------
    frames : tf.Tensor
        shape : (N,H,W,C)
    """
    total_patch_num = tf.shape(patches)[0]
    patch_h, patch_w = patch_size
    patch_c = tf.shape(patches)[3]
    frame_h= frame_size[0]
    frame_w= frame_size[1]
    
    valid_h = (patch_h-2*overlap)

    need_extra_pad_h = int((frame_h)%valid_h > 0)
    patch_num_h = (frame_h) // valid_h
    reshape_h = need_extra_pad_h + patch_num_h
    left_over_h = need_extra_pad_h * ((frame_h)%valid_h)

    valid_w = (patch_w-2*overlap)

    need_extra_pad_w = int((frame_w)%valid_w > 0)
    patch_num_w = (frame_w) // valid_w
    reshape_w = need_extra_pad_w + patch_num_w
    left_over_w = need_extra_pad_w * ((frame_w)%valid_w)


    N = total_patch_num // (reshape_h * reshape_w)
    patches = tf.reshape( patches,
        (reshape_h, reshape_w, N, patch_h, patch_w, patch_c)
    )
    patches = tf.transpose(
        patches,
        (2,0,1,3,4,5)
    )

    minimal_frame_patches = patches[
        :,
        :patch_num_h,
        :patch_num_w,
        overlap:-overlap,
        overlap:-overlap
    ]
    minimal_frame_patches_t = tf.transpose(
        minimal_frame_patches,
        (0,1,3,2,4,5)
    )
    mf_shape = tf.shape(minimal_frame_patches_t)
    new_shape=tf.concat([
        mf_shape[:1],
        [tf.reduce_prod(mf_shape[1:3])],
        [tf.reduce_prod(mf_shape[3:5])],
        mf_shape[5:]
    ],axis=0)
    minimal_frame = tf.reshape(minimal_frame_patches_t,new_shape)

    if left_over_h > 0:
        left_over_row_patches = patches[
            :,
            -1,
            :patch_num_w,
            overlap:overlap+left_over_h,
            overlap:-overlap,
        ]
        left_over_row_patches_t = tf.transpose(
            left_over_row_patches,
            (0,2,1,3,4)
        )
        lor_shape = tf.shape(left_over_row_patches_t)
        new_shape = tf.concat([
            lor_shape[:2],
            [-1],
            lor_shape[4:]
        ], axis=0)
        left_over_row = tf.reshape(left_over_row_patches_t,new_shape)
        row_added_frame = tf.concat([
            minimal_frame,
            left_over_row,
        ], axis=1)
    else:
        row_added_frame = minimal_frame
    
    if left_over_w > 0:
        left_over_col_patches = patches[
            :,
            :patch_num_h,
            -1,
            overlap:-overlap,
            overlap:overlap+left_over_w,
        ]
        
        loc_shape = tf.shape(left_over_col_patches)
        new_shape = tf.concat([
            loc_shape[:1],
            [-1],
            loc_shape[3:]
        ], axis=0)
        left_over_col = tf.reshape(left_over_col_patches,new_shape)

        if left_over_h > 0:
            left_over_col=tf.concat([
                left_over_col,
                patches[
                :,-1,-1,overlap:overlap+left_over_h,overlap:overlap+left_over_w
            ]],axis=1)
        frames = tf.concat([
            row_added_frame,
            left_over_col
        ], axis=2)

    else:
        frames = row_added_frame
    return frames


if __name__ == '__main__':
    import imageio
    from matplotlib import pyplot as plt
    import numpy as np
    t = np.array([imageio.imread('test.png')]*4)
    patches = frame_to_patch_on_batch(t, (70,80), 16)
    tp = patch_to_frame_on_batch(patches, (70,80),t.shape[1:3], 16)
    print(tf.reduce_all(t==tp))
    plt.imshow(tp.numpy()[0])
    plt.show()

    