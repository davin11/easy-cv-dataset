import os
import pandas
from multiprocessing.pool import ThreadPool
import tensorflow as tf

def _index_directory(
    directory,
    labels,
    formats,
    class_names=None,
    follow_links=False,
):
    """List all files in `directory`, with their labels.

    Args:
        directory: Directory where the data is located.
            If `labels` is `"inferred"`, it should contain
            subdirectories, each containing files for a class.
            Otherwise, the directory structure is ignored.
        labels: Either `"inferred"`
            (labels are generated from the directory structure),
            `None` (no labels),
            or a list/tuple of integer labels of the same size as the number
            of valid files found in the directory.
            Labels should be sorted according
            to the alphanumeric order of the image file paths
            (obtained via `os.walk(directory)` in Python).
        formats: Allow list of file extensions to index
            (e.g. `".jpg"`, `".txt"`).
        class_names: Only valid if `labels="inferred"`. This is the explicit
            list of class names (must match names of subdirectories). Used
            to control the order of the classes
            (otherwise alphanumeric order is used).
        follow_links: Whether to visits subdirectories pointed to by symlinks.

    Returns:
        tuple (file_paths, labels, class_names).
        - file_paths: list of file paths (strings).
        - labels: list of matching integer labels (same length as file_paths)
        - class_names: names of the classes corresponding to these labels, in
        order.
    """
    if labels == "inferred":
        subdirs = []
        for subdir in sorted(tf.io.gfile.listdir(directory)):
            if tf.io.gfile.isdir(tf.io.gfile.join(directory, subdir)):
                if not subdir.startswith("."):
                    if subdir.endswith("/"):
                        subdir = subdir[:-1]
                    subdirs.append(subdir)
        if class_names is not None:
            if not set(class_names).issubset(set(subdirs)):
                raise ValueError(
                    "The `class_names` passed did not match the "
                    "names of the subdirectories of the target directory. "
                    f"Expected: {subdirs} (or a subset of it), "
                    f"but received: class_names={class_names}"
                )
            subdirs = class_names  # Keep provided order.
    else:
        # In the explicit/no-label cases, index from the parent directory down.
        subdirs = [""]
        if class_names is not None:
            if labels is None:
                raise ValueError(
                    "When `labels=None` (no labels), argument `class_names` "
                    "cannot be specified."
                )
            else:
                raise ValueError(
                    "When argument `labels` is specified, argument "
                    "`class_names` cannot be specified (the `class_names` "
                    "will be the sorted list of labels)."
                )
    class_names = subdirs
    class_indices = dict(zip(class_names, range(len(class_names))))

    # Build an index of the files
    # in the different class subfolders.
    pool = ThreadPool()
    results = []
    filenames = []

    for dirpath in (tf.io.gfile.join(directory, subdir) for subdir in subdirs):
        results.append(
            pool.apply_async(
                _index_subdirectory,
                (dirpath, class_indices, follow_links, formats),
            )
        )
    labels_list = []
    for res in results:
        partial_filenames, partial_labels = res.get()
        labels_list.append(list(partial_labels))
        filenames += partial_filenames

    if labels == "inferred":
        # Inferred labels.
        labels = sum(labels_list, list())
    elif labels is None:
        class_names = None
    else:
        # Manual labels.
        if len(labels) != len(filenames):
            raise ValueError(
                "Expected the lengths of `labels` to match the number "
                "of files in the target directory. len(labels) is "
                f"{len(labels)} while we found {len(filenames)} files "
                f"in directory {directory}."
            )
        class_names = [str(label) for label in sorted(set(labels))]
    pool.close()
    pool.join()
    file_paths = [tf.io.gfile.join(directory, filename) for filename in filenames]

    return file_paths, labels, class_names


def _iter_valid_files(directory, follow_links, formats):
    if not follow_links:
        walk = tf.io.gfile.walk(directory)
    else:
        walk = os.walk(directory, followlinks=follow_links)
    for root, _, files in sorted(walk, key=lambda x: x[0]):
        for fname in sorted(files):
            if fname.lower().endswith(formats):
                yield root, fname


def _index_subdirectory(directory, class_indices, follow_links, formats):
    """Recursively walks directory and list image paths and their class index.

    Args:
        directory: string, target directory.
        class_indices: dict mapping class names to their index.
        follow_links: boolean, whether to recursively follow subdirectories
            (if False, we only list top-level images in `directory`).
        formats: Allow list of file extensions to index (e.g. ".jpg", ".txt").

    Returns:
        tuple `(filenames, labels)`. `filenames` is a list of relative file
            paths, and `labels` is a list of integer labels corresponding
            to these files.
    """
    dirname = os.path.basename(directory)
    valid_files = _iter_valid_files(directory, follow_links, formats)
    labels = []
    filenames = []
    for root, fname in valid_files:
        labels.append(class_indices[dirname])
        absolute_path = tf.io.gfile.join(root, fname)
        relative_path = tf.io.gfile.join(
            dirname, os.path.relpath(absolute_path, directory)
        )
        filenames.append(relative_path)
    return filenames, labels

def dataframe_from_directory(directory, formats, colname_file, colname_class, follow_links=False):
    """Generates a `pandas.DataFrame` from files in a directory.

    If your directory structure is:

    ```
    main_directory/
    ...class_a/
    ......file_1.ext
    ......file_2.ext
    ...class_b/
    ......file_5.ext
    ......file_6.ext
    ```

    Then calling `dataframe_from_directory(main_directory)` will
    return a `pandas.DataFrame` that yields batches of
    images from the subdirectories `class_a` and `class_b`.

    Args:
      directory: Directory where the data is located.
      formats: Allow list of file extensions to index
            (e.g. `".jpg"`, `".txt"`).
      follow_links: Whether to visit subdirectories pointed to by symlinks.
          Defaults to False.
      colname_file: column name for image file. Default: 'image'.
      colname_class:: column name for class. Default: 'class'.
    Returns:
      A `pandas.DataFrame` object.

    """
    image_paths, labels, class_names = _index_directory(directory, labels="inferred", formats=formats,
                                                        follow_links=follow_links)
    dataframe = pandas.DataFrame(
        {colname_file: image_paths,
         colname_class: [class_names[_] for _ in labels]}
    )
    return dataframe

def dataset_from_dataframe(
    dataframe,
    colname_input,
    colname_target,
    load_fun_input,
    load_fun_target,
    shuffle=False,
    seed=None,
    pre_batching_operation=None,
    batch_size=None,
    buffer_shuffle_size=None,
    post_batching_operation=None,
    dictname_input=None,
    dictname_target=None,
):
    if shuffle:
        assert seed is not None
    if dictname_input is None:
        dictname_input = colname_input
    if dictname_target is None:
        dictname_target = colname_target

    if isinstance(dataframe, tf.data.Dataset):
        dataset = dataframe
        num_elements = dataframe.cardinality()
    else:
        if shuffle:
            dataframe = dataframe.sample(frac=1, random_state=seed)
        dataset = tf.data.Dataset.from_tensor_slices(
            {
                colname_input: dataframe[colname_input],
                colname_target: dataframe[colname_target],
            }
        )
        num_elements = len(dataframe[colname_input])

    if shuffle:
        if buffer_shuffle_size is None:
            if batch_size is not None:
                buffer_shuffle_size = 10 * batch_size
            elif num_elements > 3:
                buffer_shuffle_size = num_elements // 4
            else:
                buffer_shuffle_size = 1024
        print("shuffling with buffer_size %d" % buffer_shuffle_size)
        dataset = dataset.shuffle(
            buffer_size=buffer_shuffle_size,
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda x: {
            dictname_input: load_fun_input(x[colname_input]),
            dictname_target: load_fun_target(x[colname_target]),
        }
    )

    if pre_batching_operation is not None:
        dataset = dataset.map(
            pre_batching_operation, num_parallel_calls=tf.data.AUTOTUNE
        )

    if batch_size is not None:
        #try:
        dataset = dataset.ragged_batch(batch_size)
        #    # dataset = dataset.batch(batch_size)
        #except:
        #    dataset = dataset.apply(
        #        tf.data.experimental.dense_to_ragged_batch(batch_size)
        #    )

    if post_batching_operation is not None:
        dataset = dataset.map(
            post_batching_operation, num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset