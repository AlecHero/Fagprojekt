import numpy as np
import nibabel as nib
import pandas as pd
import h5py
from scipy.signal import butter, filtfilt, hilbert
import os
# filter stuff, repetition time is 2.1s
nyquist = 0.5 * 1/2.1
low = 0.008 / nyquist
high = 0.09 / nyquist
b, a = butter(2, [low, high], btype='band')

# load the atlas
atlas = nib.load('Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1_3T_MNI152NLin2009cAsym_2mm.nii.gz')
atlas_data = atlas.get_fdata()
atlas_data = np.round(atlas_data)
atlas_labels = np.unique(atlas_data)
atlas_labels = atlas_labels[1:] # remove 0
atlas_labels = atlas_labels.astype(int)

# load the subjects
for i in range(33):
    # load sleep-score tsv file
    sleep_score = pd.read_csv('sleep_scores/sub-'+str(i+1).zfill(2)+'-sleep-stage.tsv', sep='\t')
    sessions = sleep_score['session'].unique()
    for session in sessions:
        # if cohmap file exists continue
        if os.path.exists('cohmaps/sub-'+str(i+1).zfill(2)+'_session-'+session+'_coh.h5'):
            continue
        print('Processing subject '+str(i+1).zfill(2)+' session '+session+' ...')
        #get data:
        data = nib.load('ds003768-fmriprep/sub-'+str(i+1).zfill(2)+'/func/sub-'+str(i+1).zfill(2)+'_'+session+'_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')
        data = data.get_fdata()

        # parcellate the data
        print('Parcellating data ...')
        data_region = np.zeros((data.shape[3],len(atlas_labels)))
        for j in range(len(atlas_labels)):
            for k in range(data.shape[3]):
                data_region[k,j] = np.mean(data[:,:,:,k][atlas_data == atlas_labels[j]])

        # filter the data
        print('Filtering data ...')
        data_region_filtered = np.zeros(data_region.shape)
        for j in range(len(atlas_labels)):
            data_region_filtered[:,j] = filtfilt(b, a, data_region[:,j])
        
        print('Computing Hilbert phase')
        data_region_phase = np.zeros(data_region_filtered.shape)
        for j in range(len(atlas_labels)):
            data_region_phase[:,j] = np.angle(hilbert(data_region_filtered[:,j]))

        print('Computing coherence maps and eigenvectors/eigenvalues ...')
        all_coh = np.zeros((data_region_phase.shape[0],data_region_phase.shape[1],data_region_phase.shape[1]))
        all_eigvecs = np.zeros((data_region_phase.shape[0],data_region_phase.shape[1],2))
        all_eigvals = np.zeros((data_region_phase.shape[0],2))
        for t in range(data_region_phase.shape[0]):
            coh_map = np.outer(np.cos(data_region_phase[t]),np.cos(data_region_phase[t]))+np.outer(np.sin(data_region_phase[t]),np.sin(data_region_phase[t]))
            all_coh[t] = coh_map
            eigval,eigvec = np.linalg.eigh(coh_map)
            all_eigvecs[t] = eigvec[:,-2:]
            all_eigvals[t] = eigval[-2:]
        # save all as h5
        with h5py.File('cohmaps/sub-'+str(i+1).zfill(2)+'_session-'+session+'_coh.h5', 'w') as f:
            f.create_dataset('coh', data=all_coh)
        with h5py.File('eigvecs/sub-'+str(i+1).zfill(2)+'_session-'+session+'_eigvecs.h5', 'w') as f:
            f.create_dataset('eigvecs', data=all_eigvecs)
        with h5py.File('eigvals/sub-'+str(i+1).zfill(2)+'_session-'+session+'_eigvals.h5', 'w') as f:
            f.create_dataset('eigvals', data=all_eigvals)