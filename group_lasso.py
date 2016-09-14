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
from sklearn import linear_model

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# if want to output log to a file instead of outputting log to the console,
# replace "stream=sys.stdout" with "filename='fcma.log'"
logging.basicConfig(level=logging.INFO, format=format, stream=sys.stdout)
logger = logging.getLogger(__name__)
MAX_ITER = 1000

def separateEpochs(activity_data, activity_data2, epoch_list):
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
    epoch_list: list of 3D array in shape [condition, nEpochs, nTRs]
        specification of epochs and conditions
        assuming all subjects have the same number of epochs
        len(epoch_list) equals the number of subjects

    Returns
    -------
    raw_data: list of 2D array in shape [epoch length, nVoxels]
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
    for sid in range(len(epoch_list)):
        epoch = epoch_list[sid]
        # avg_per_subj is in shape [n_epochs_per_subj, n_voxels]
        avg_per_subj = np.zeros([epoch.shape[1], n_voxels], np.float32, order='C')
        count = 0
        for cond in range(epoch.shape[0]):
            sub_epoch = epoch[cond, :, :]
            for eid in range(epoch.shape[1]):
                r = np.sum(sub_epoch[eid, :])
                if r > 0:   # there is an epoch in this condition
                    # mat is row-major, in shape [epoch_length, n_voxels]
                    # regardless of the order of acitvity_data[sid]
                    mat = activity_data[sid][sub_epoch[eid, :] == 1, :]
                    mat2 = activity_data2[sid][sub_epoch[eid, :] == 1, :]
                    avg_per_subj[count, :] = np.copy(np.mean(mat2, axis=0))
                    count += 1
                    mat = zscore(mat, axis=0, ddof=0)
                    # if zscore fails (standard deviation is zero),
                    # set all values to be zero
                    mat = np.nan_to_num(mat)
                    mat = mat / math.sqrt(r)
                    raw_data.append(mat)
                    labels.append(cond)
        assert count == epoch.shape[1], \
            'subject %d does not have right number of epochs, %d %d' % (sid, count, epoch.shape[1])
        avg_per_subj = zscore(avg_per_subj, axis=0, ddof=0)
        for i in range(avg_per_subj.shape[0]):
            avg_data.append(avg_per_subj[i, :])
    assert len(raw_data) == len(avg_data), \
        'either raw_data or avg_data does not have right epochs'
    time2 = time.time()
    logger.debug(
        'epoch separation done, takes %.2f s' %
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
                          epoch_file, n_tops):
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

    epoch_list = np.load(epoch_file)
    raw_data, avg_data, labels = separateEpochs(activity_data, activity_data2, epoch_list)

    # feature_vectors is in shape [n_epochs, n_tops*n_tops+n_tops]
    feature_vectors = np.zeros([len(raw_data), n_tops*n_tops+n_tops], np.float32, order='C')
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
                             0.0, feature_vectors,
                             n_tops, count)
        count += 1
    #np.save('corr_feature_vectors', feature_vectors)
    # activity features
    #feature_vectors = getSearchlightFeatures(data, corr_masked_seq, feature_vectors)
    for i in range(feature_vectors.shape[0]):
        feature_vectors[i, n_tops*n_tops: n_tops*n_tops+n_tops] = np.copy(avg_data[i])

    feature_vectors, labels = map(np.asanyarray, (feature_vectors, labels))
    return feature_vectors, labels, n_subjs

def group_lasso(X, y, alpha, groups, max_iter=MAX_ITER, rtol=1e-6,
                verbose=False):
    """
    Linear least-squares with l2/l1 regularization solver.
    Solves problem of the form:
               .5 * |Xb - y| + n_samples * alpha * Sum(w_j * |b_j|)
    where |.| is the l2-norm and b_j is the coefficients of b in the
    j-th group. This is commonly known as the `group lasso`.
    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Design Matrix.
    y : array of shape (n_samples,)
    alpha : float or array
        Amount of penalization to use.
    groups : array of shape (n_features,)
        Group label. For each column, it indicates
        its group apertenance.
    rtol : float
        Relative tolerance. ensures ||(x - x_) / x_|| < rtol,
        where x_ is the approximate solution and x is the
        true solution.
    Returns
    -------
    x : array
        vector of coefficients
    References
    ----------
    "Efficient Block-coordinate Descent Algorithms for the Group Lasso",
    Qin, Scheninberg, Goldfarb
    """

    # .. local variables ..
    X, y, groups, alpha = map(np.asanyarray, (X, y, groups, alpha))
    if len(groups) != X.shape[1]:
        raise ValueError("Incorrect shape for groups")
    w_new = np.zeros(X.shape[1], dtype=X.dtype)
    alpha = alpha * X.shape[0]

    # .. use integer indices for groups ..
    group_labels = [np.where(groups == i)[0] for i in np.unique(groups)]
    H_groups = [np.dot(X[:, g].T, X[:, g]) for g in group_labels]
    eig = list(map(linalg.eigh, H_groups))
    Xy = np.dot(X.T, y)
    initial_guess = np.zeros(len(group_labels))

    def f(x, qp2, eigvals, alpha):
        return 1 - np.sum( qp2 / ((x * eigvals + alpha) ** 2))
    def df(x, qp2, eigvals, penalty):
        # .. first derivative ..
        return np.sum((2 * qp2 * eigvals) / ((penalty + x * eigvals) ** 3))

    if X.shape[0] > X.shape[1]:
        H = np.dot(X.T, X)
    else:
        H = None

    for n_iter in range(max_iter):
        w_old = w_new.copy()
        for i, g in enumerate(group_labels):
            # .. shrinkage operator ..
            eigvals, eigvects = eig[i]
            w_i = w_new.copy()
            w_i[g] = 0.
            if H is not None:
                X_residual = np.dot(H[g], w_i) - Xy[g]
                #print(H[g].shape, w_i.shape, X_residual.shape)
            else:
                #print(X.T.shape, X[:, g].shape, w_i.shape)
                X_residual = np.dot(X.T[g, :], np.dot(X, w_i)) - Xy[g]
            qp = np.dot(eigvects.T, X_residual)
            if len(g) < 2:
                # for single groups we know a closed form solution
                w_new[g] = - np.sign(X_residual) * max(abs(X_residual) - alpha, 0)
            else:
                if alpha < linalg.norm(X_residual, 2):
                    initial_guess[i] = optimize.newton(f, initial_guess[i], df, tol=.5,
                                                       args=(qp ** 2, eigvals, alpha))
                    w_new[g] = - initial_guess[i] * np.dot(eigvects /  (eigvals * initial_guess[i] + alpha), qp)
                else:
                    w_new[g] = 0.


        # .. dual gap ..
        #max_inc = linalg.norm(w_old - w_new, np.inf)
        if True: #max_inc < rtol * np.amax(w_new):
            residual = np.dot(X, w_new) - y
            group_norm = alpha * np.sum([linalg.norm(w_new[g], 2)
                                         for g in group_labels])
            if H is not None:
                norm_Anu = [linalg.norm(np.dot(H[g], w_new) - Xy[g]) \
                            for g in group_labels]
            else:
                norm_Anu = [linalg.norm(np.dot(X.T[g, :], np.dot(X, w_new)) - Xy[g]) \
                            for g in group_labels]
            if np.any(norm_Anu > alpha):
                nnu = residual * np.min(alpha / norm_Anu)
            else:
                nnu = residual
            primal_obj =  .5 * np.dot(residual, residual) + group_norm
            dual_obj   = -.5 * np.dot(nnu, nnu) - np.dot(nnu, y)
            dual_gap = primal_obj - dual_obj
            if verbose:
                print('Relative error:', dual_gap / dual_obj)
            if np.abs(dual_gap / dual_obj) < rtol:
                break

    logger.info(
        'Group LASSO done, iteration times: %d' % n_iter
    )

    return w_new

# python group_lasso.py ~/data/face_scene/raw/ nii.gz ~/data/face_scene/results/corr/sub18_seq.nii.gz ~/data/face_scene/results/acti/sub18_seq.nii.gz ~/IdeaProjects/brainiak/examples/fcma/data/fs_epoch_labels.npy 10
if __name__ == '__main__':
    time1 = time.time()
    n_tops = int(sys.argv[6])
    # left_out is 0-based subject id
    left_out = int(sys.argv[7])
    feature_vectors, labels, n_subjs = prepareFeatureVectors(sys.argv[1], sys.argv[2], sys.argv[3],
                                 sys.argv[4], sys.argv[5], n_tops)
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
    X = np.concatenate((feature_vectors[0: start_test_sample, :], feature_vectors[end_test_sample:, :]))
    y = np.concatenate((labels[0: start_test_sample], labels[end_test_sample:]))
    X = zscore(X, axis=0, ddof=0)
    X = X / math.sqrt(X.shape[0])
    #coef = group_lasso(X, y, 0.01, groups)
    clf = linear_model.Lasso(alpha=0.005)
    clf.fit(X, y)
    logger.info(
        '%d features have been selected from %d features (%d correlation + %d activity), '
        'of which %d are from correlation, %d are from activity' %
        (len(np.where(clf.coef_!=0)[0]), clf.coef_.shape[0],
         n_tops*n_tops, n_tops,
         len(np.where(clf.coef_[0:n_tops*n_tops]!=0)[0]), len(np.where(clf.coef_[n_tops*n_tops:]!=0)[0]))
    )
    predict_results = np.sign(clf.predict(feature_vectors[start_test_sample:end_test_sample, :]))
    n_correct = np.count_nonzero(predict_results == labels[start_test_sample:end_test_sample])
    logger.info(
        'prediction accuracy: %d/%d=%.2f%%' %
        (n_correct, len(predict_results), n_correct*100.0/len(predict_results))
    )
    time2 = time.time()
    logger.info(
        'LASSO done, takes %.2f s' %
        (time2 - time1)
    )
    #np.save('lasso_coef', clf.coef_)
