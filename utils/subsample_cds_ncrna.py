def subsample_cds_ncrna(filepath, N=10**7):
    """subsamples the cds or ncrna files based on specified sizes"""
    with open(filepath, 'r') as f:
        text = f.read()
    text = ''.join([i if re.sub('[^ACGT]', '', i) == i else ' ' for i in text.split('\n')])
    cdss = text.split()
    
    subsampled_cdss = []
    subsample_length = 0
    np.random.seed(1872)
    randomized_indices = np.random.choice(range(len(cdss)), len(cdss), replace=False)

    for k in randomized_indices:
        c = cdss[k]
        subsample_length += len(c)
        subsampled_cdss.append(c)
        if subsample_length >= N:
            break
            
    return subsampled_cdss
