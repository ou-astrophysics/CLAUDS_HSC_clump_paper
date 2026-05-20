#!/usr/bin/env python

# [START all]
# [START libraries]
import pandas as pd
import numpy as np
import os
import sys
import glob
import argparse
from tqdm import tqdm
import sys
sys.path.append('./shared/')
import GalaxyMeasurements
import prospector_utils
from prospect.io import read_results as reader
# [END libraries]


# [START initial settings]
# [END initial settings]


# [START classes and functions]
# [END classes and functions]


# [START Main]
def main():
    # Parse arguments from cmd
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', dest='batch', help='Batch to process', type=str)

    args = parser.parse_args()
    batch = args.batch

    # load data
    print('Loading data...')
    chunks = 20
    _df_list = []
    
    for i in range(chunks):
        _df = pd.read_parquet('/data/objects_for_sed_fitting_batch_{}_chunk_{}.gzip'.format(batch, i), engine='pyarrow')
        _df_list.append(_df)
    
    df = pd.concat(_df_list)
    
    df_list = []
    error_list = []
    
    for idx, data in tqdm(df.iterrows(), total=df.shape[0]):
        objid = data['HSCobjid']
        clump_id = data['clump_id']
        peak_id = data['peak_id']
    
        hfile = '/sed_fits/' + str(objid)[-3:] + '/' + 'sed_fit_{}_{}_{}.h5'.format(objid, clump_id, peak_id)
    
        if os.path.exists(hfile) and os.path.getsize(hfile)>0:
            result, obs, model = reader.results_from(hfile, dangerous=False)
            
            fit_result_dict = prospector_utils.collect_results(result, bands='GRIZY', logify=['mass'])
            df_list.append(pd.DataFrame.from_dict([fit_result_dict]))
    
        else:
            # print("Can't find a SED-fit for {}-{}-{}".format(objid, clump_id, peak_id))
            error_list.append([objid, clump_id, peak_id])
    
    df_result = pd.concat(df_list)
    
    df_result.to_parquet('sed_fits_result_batch_{}.gzip'.format(batch), compression='gzip')
    
    print(error_list)
# [END Main]


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit(1)
# [END all]
