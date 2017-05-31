import pyximport
pyximport.install()
import sys
import nibabel as nib
import os
import math
import time
import numpy as np
import logging
from scipy.stats.mstats import zscore
import cython_blas as blas
from scipy import linalg, optimize
from sklearn import svm

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# if want to output log to a file instead of outputting log to the console,
# replace "stream=sys.stdout" with "filename='fcma.log'"
logging.basicConfig(level=logging.INFO, format=format, stream=sys.stdout)
logger = logging.getLogger(__name__)
MAX_ITER = 1000

def separateEpochs(activity_data, activity_data2, corr_epoch_list, acti_epoch_list):
    """ separate data into epochs of interest specified in epoch_list
    and z-score them for computing correlation

    Parameters
    ----------
    activity_data: list of 2D array in shape [nTRs, nVoxels]
        the masked activity data organized in TR*voxel formats of all subjects
        for correlation
    activity_data2: list of 2D array in shape [nTRs, nVoxels]
        the masked activity data organized in TR*voxel formats of all subjects
        for activity
    corr_epoch_list: list of 3D array in shape [condition, nEpochs, nTRs]
        specification of epochs and conditions
        assuming all subjects have the same number of epochs
        len(corr_epoch_list) equals the number of subjects
    acti_epoch_list: list of 3D array in shape [condition, nEpochs, nTRs]
        specification of epochs and conditions
        assuming all subjects have the same number of epochs
        len(acti_epoch_list) equals the number of subjects

    Returns
    -------
    raw_data: list of 2D array in shape [corr_epoch length, nVoxels]
        the data organized in epochs
        and z-scored in preparation of correlation computation
        len(raw_data) equals the number of epochs
    avg_data: list of 1D array in shape [nVoxels,]
        normalized average data over epochs
        len(avg_data) equals the number of epochs
    labels: list of 1D array
        the condition labels of the epochs
        len(labels) labels equals the number of epochs
    """
    time1 = time.time()
    raw_data = []
    avg_data = []
    labels = []
    n_voxels = activity_data[0].shape[1]
    for sid in range(len(corr_epoch_list)):
        corr_epoch = corr_epoch_list[sid]
        acti_epoch = acti_epoch_list[sid]
        # avg_per_subj is in shape [n_epochs_per_subj, n_voxels]
        avg_per_subj = np.zeros([acti_epoch.shape[1], n_voxels], np.float32, order='C')
        count = 0
        for cond in range(corr_epoch.shape[0]):
            corr_sub_epoch = corr_epoch[cond, :, :]
            acti_sub_epoch = acti_epoch[cond, :, :]
            for eid in range(corr_epoch.shape[1]):
                # correlation epochs
                r = np.sum(corr_sub_epoch[eid, :])
                if r > 0:   # there is an corr_epoch in this condition
                    # mat is row-major, in shape [epoch_length, n_voxels]
                    # regardless of the order of acitvity_data[sid]
                    mat = activity_data[sid][corr_sub_epoch[eid, :] == 1, :]
                    mat = zscore(mat, axis=0, ddof=0)
                    # if zscore fails (standard deviation is zero),
                    # set all values to be zero
                    mat = np.nan_to_num(mat)
                    mat = mat / math.sqrt(r)
                    raw_data.append(mat)
                    labels.append(cond)
                # activity epochs
                r = np.sum(acti_sub_epoch[eid, :])
                if r > 0:
                    mat2 = activity_data2[sid][acti_sub_epoch[eid, :] == 1, :]
                    avg_per_subj[count, :] = np.copy(np.mean(mat2, axis=0))
                    count += 1
        assert count == corr_epoch.shape[1], \
            'subject %d does not have right number of epochs, %d %d' % (sid, count, corr_epoch.shape[1])
        avg_per_subj = zscore(avg_per_subj, axis=0, ddof=0)
        for i in range(avg_per_subj.shape[0]):
            avg_data.append(avg_per_subj[i, :])
    assert len(raw_data) == len(avg_data), \
        'either raw_data or avg_data does not have right epochs'
    time2 = time.time()
    logger.debug(
        'corr_epoch separation done, takes %.2f s' %
        (time2 - time1)
    )
    return raw_data, avg_data, labels

#def getSearchlightFeatures(data, masked_seq, feature_vectors):
#    return feature_vectors

def generateMaskedSeq(seq_file):
    seq_img = nib.load(seq_file)
    seq = seq_img.get_data()
    count = len(np.where(seq>0)[0])
    # masked_seq is organized in reversed index
    masked_seq = [None] * count
    for index in np.ndindex(seq.shape):
        if seq[index] != 0:
            masked_seq[seq[index]-1] = index
    return masked_seq

def prepareFeatureVectors(data_dir, file_extension, corr_seq_file, acti_seq_file,
                          corr_epoch_file, acti_epoch_file, n_tops):
    corr_masked_seq = generateMaskedSeq(corr_seq_file)
    acti_masked_seq = generateMaskedSeq(acti_seq_file)

    files = [f for f in sorted(os.listdir(data_dir))
             if os.path.isfile(os.path.join(data_dir, f))
             and f.endswith(file_extension)]
    n_subjs = len(files)
    activity_data = []
    activity_data2 = []
    for f in files:
        img = nib.load(os.path.join(data_dir, f))
        data = img.get_data()
        (d1, d2, d3, d4) = data.shape
        selected_data = np.zeros([d4, n_tops], np.float32, order='C')
        count1 = 0
        for index in corr_masked_seq[0: n_tops]:
            selected_data[:, count1] = np.copy(data[index])
            count1 += 1
        activity_data.append(selected_data)
        selected_data2 = np.zeros([d4, n_tops], np.float32, order='C')
        count1 = 0
        for index in acti_masked_seq[0: n_tops]:
            selected_data2[:, count1] = np.copy(data[index])
            count1 += 1
        activity_data2.append(selected_data2)
        logger.debug(
            'file %s is loaded and top-%d correlaton and acitivity voxels are selected, '
            'with data shape %s' %
            (f, n_tops, selected_data.shape)
        )
    #np.save('activity_data', activity_data)
    #activity_data = np.load('activity_data.npy')

    corr_epoch_list = np.load(corr_epoch_file)
    acti_epoch_list = np.load(acti_epoch_file)
    raw_data, avg_data, labels = separateEpochs(activity_data, activity_data2, corr_epoch_list, acti_epoch_list)

    # feature_vectors is in shape [n_epochs, n_tops*(n_tops-1)/2+n_tops]
    num_corr_features = int(n_tops*(n_tops-1)/2)
    feature_vectors = np.zeros([len(raw_data), num_corr_features+n_tops], np.float32, order='C')
    corr_buf = np.zeros([n_tops, n_tops], np.float32, order='C')
    # correlation features
    count = 0
    for multiplier in raw_data:
        # multiplier is in fortran order (column-major) no matter of what rd is in
        no_trans = 'N'
        trans = 'T'
        blas.compute_correlation(no_trans, trans,
                             n_tops, n_tops,
                             multiplier.shape[0], 1.0,
                             multiplier, n_tops,
                             multiplier, n_tops,
                             0.0, corr_buf,
                             n_tops, 0)
        count1 = 0
        for i in range(n_tops):
            feature_vectors[count, count1:count1+i] = corr_buf[i, 0:i]
            count1 += i
        count += 1
    #np.save('corr_feature_vectors', feature_vectors)
    # activity features
    #feature_vectors = getSearchlightFeatures(data, corr_masked_seq, feature_vectors)
    for i in range(feature_vectors.shape[0]):
        feature_vectors[i, num_corr_features: num_corr_features+n_tops] = np.copy(avg_data[i])

    feature_vectors, labels = map(np.asanyarray, (feature_vectors, labels))
    return feature_vectors, labels, n_subjs

def compute_kernel_matrix_in_portion(X, num_unit_features=1000):
    num_samples = X.shape[0]
    num_features = X.shape[1]
    sr = 0
    processed_features = num_unit_features
    kernel_matrix = np.zeros([num_samples, num_samples])
    while sr < num_features:
        if processed_features > num_features - sr:
            processed_features = num_features - sr
        sub_x = X[:, sr: processed_features]
        kernel_matrix += np.dot(sub_x, sub_x.transpose())
        sr += num_unit_features
    return kernel_matrix


# python group_svm.py ~/data/face_scene/raw/ nii.gz ~/data/face_scene/results/corr/sub18_seq.nii.gz ~/data/face_scene/results/acti/sub18_seq.nii.gz ~/IdeaProjects/brainiak/examples/fcma/face_scene/fs_epoch_labels.npy ~/IdeaProjects/brainiak/examples/fcma/face_scene/fs_epoch_labels.npy 10 17
if __name__ == '__main__':
    time1 = time.time()
    n_tops = int(sys.argv[7])
    # left_out is 0-based subject id
    left_out = int(sys.argv[8])
    feature_vectors, labels, n_subjs = prepareFeatureVectors(sys.argv[1], sys.argv[2], sys.argv[3],
                                 sys.argv[4], sys.argv[5], sys.argv[6], n_tops)
    time2 = time.time()
    logger.info(
        'feature vector preparation done, takes %.2f s' %
        (time2 - time1)
    )
    time1 = time.time()
    #groups = np.zeros((n_tops, n_tops), np.int)
    #for i in range(n_tops):
    #    groups[i, :] = i
    #groups = groups.reshape(n_tops * n_tops)
    #groups = np.zeros(n_tops * n_tops, np,int)
    #for i in range(n_tops*n_tops):
    #    groups[i] = i
    labels[labels==0] = -1
    n_per_subjs = int(len(labels) / n_subjs)
    start_test_sample = n_per_subjs * left_out
    end_test_sample = start_test_sample + n_per_subjs
#    feature_vectors = zscore(feature_vectors, axis=0, ddof=0)
#    feature_vectors = feature_vectors / math.sqrt(feature_vectors.shape[0])
    X = np.concatenate((feature_vectors[0: start_test_sample, :], feature_vectors[end_test_sample:, :]))
    y = np.concatenate((labels[0: start_test_sample], labels[end_test_sample:]))
    X = zscore(X, axis=0, ddof=0)
    X = X / math.sqrt(X.shape[0])
    kernel_matrix = compute_kernel_matrix_in_portion(X)
    #coef = group_lasso(X, y, 0.01, groups)
    #clf = linear_model.Lasso(alpha=0.005)
    clf = svm.SVC(kernel='precomputed', shrinking=False, C=1)
#    clf = linear_model.LassoCV(max_iter=10000, n_jobs=-1)
    clf.fit(kernel_matrix, y)
#    logger.info(
#        '%d features have been selected from %d features (%d correlation + %d activity), '
#        'of which %d are from correlation, %d are from activity. chosen alpha %f' %
#        (len(np.where(clf.coef_!=0)[0]), clf.coef_.shape[0],
#         n_tops*n_tops, n_tops,
#         len(np.where(clf.coef_[0:n_tops*n_tops]!=0)[0]), len(np.where(clf.coef_[n_tops*n_tops:]!=0)[0]),
#         clf.alpha_)
#    )
    test_vector = feature_vectors[start_test_sample:end_test_sample, :]
    test_vector = zscore(test_vector, axis=0, ddof=0)
    test_vector = test_vector / math.sqrt(test_vector.shape[0])
    test_feature_vector = np.dot(test_vector, X.transpose())
    predict_results = clf.predict(test_feature_vector)
#    predict_results = np.sign(predict_weights)
    n_correct = np.count_nonzero(predict_results == labels[start_test_sample:end_test_sample])
    time2 = time.time()
    logger.info(
        'SVM done, takes %.2f s' %
        (time2 - time1)
    )
    logger.debug(
        'predictions are %s' %
        predict_results
    )
    logger.debug(
        'ground truth labels are %s' %
        labels[start_test_sample:end_test_sample]
    )
    logger.info(
        'prediction accuracy: %d/%d=%.2f%%' %
        (n_correct, len(predict_results), n_correct*100.0/len(predict_results))
    )
    #np.save('lasso_coef', clf.coef_)
