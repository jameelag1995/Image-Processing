#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore', module ='numba')

import os, sys, traceback, subprocess, shutil, argparse
import numpy as np
from scipy.io import wavfile
import sys



def presubmit():
    print ('Ex2 Presubmission Script')
    disclaimer="""
    Disclaimer
    ----------
    The purpose of this script is to make sure that your code is compliant
    with the exercise API and some of the requirements
    The script does not test the quality of your results.
    Don't assume that passing this script will guarantee that you will get
    a high grade in the exercise \n\n
    """
    print (disclaimer)

    print('=== Check Submission ===\n')
    if not os.path.exists('current/README.md'):
        print ('No readme!')
        return False
    else:
        print('README file:\n')
        with open ('current/README.md') as f:
            print(f.read())

    for q in [1,2,3]:
        if not os.path.exists('current/answer_q%d.txt'%q):
            print ('No answer_q%d.txt!'%q)
            return False
        print ('Answer to q%d:'%q)
        with open('current/answer_q%d.txt'%q) as f:
            print (f.read())

    print('=== Load Student Library ===\n')
    print('Loading...')
    sys.stdout.flush()
    sol2 = None
    try:
        import current.sol2 as sol2

    except:
        print('Unable to import the solution.')
        return False
    print ('=== Section 1.1 ===\n')
    print ('DFT and IDFT')
    sys.stdout.flush()
    filename = 'external/monkey.jpg'
    try:
        im = sol2.read_image(filename, 1)
        dft_1d = sol2.DFT(im[:,:1])
        if not np.all(dft_1d.shape==im[:,:1].shape):
            print ('1D DFT has different dimensions than original array!')
            return False
        idft_1d = sol2.IDFT(dft_1d)
        if not np.all(idft_1d.shape==dft_1d.shape):
            print ('1D IDFT has different dimensions than original array!')
            return False
        if idft_1d.dtype != np.complex:
            print ('IDFT should return complex values!')
            return False
    except:
        print(traceback.format_exc())
        return False

    print ('=== Section 1.2 ===\n')
    print ('2D DFT and IDFT')
    sys.stdout.flush()
    try:
        dft = sol2.DFT2(im)
        if not np.all(dft.shape==im.shape):
            print ('2D DFT has different dimensions than original image!')
            return False
        idft = sol2.IDFT2(dft)
        if not np.all(idft.shape==dft.shape):
            print ('2D IDFT has different dimensions than original image!')
            return False
        if idft.dtype != np.complex:
            print ('IDFT should return complex values!')
            return False
    except:
        print(traceback.format_exc())
        return False

    print ('=== Section 2.1 ===\n')
    print('Fast foward by rate change')
    sys.stdout.flush()
    audio_filename = 'external/aria_4kHz.wav'
    sr, audio_data = wavfile.read(audio_filename)
    rate = 2

    print ('=== Section 2.2 ===\n')
    print ('Fast forward using Fourier')
    # sys.stdout.flush()
    # try:
    #     reduced_samples = sol2.change_samples(audio_filename, rate)
    #     if not np.all(reduced_samples.shape[0] == (audio_data.shape[0] / rate)):
    #         print ('the new samples shape should be: ', (audio_data.shape / rate), 'but is:', reduced_samples.shape[0])
    #     return False
    # except Exception:
    #     print(traceback.format_exc())
    #     return False

    print ('=== Section 2.3 ===\n')
    print ('Fast forward using Spectrogram')
    sys.stdout.flush()
    try:
        reduced_samples = sol2.resize_spectrogram(audio_data, rate)
        new_shape = 4960
        if not np.all(reduced_samples.shape[0] == new_shape):
            print ('the new samples shape should be: ', new_shape, 'but is:', reduced_samples.shape)
            return False
    except Exception:
        print(traceback.format_exc())
        return False

    print ('=== Section 2.4 ===\n')
    print ('Fast forward using Spectrogram and phase vocoder')
    # sys.stdout.flush()
    # try:
    #     magnitude = sol2.fourier_der(im)
    #     if not np.all(magnitude.shape == im.shape):
    #         print ('derivative magnitude shape should be :', im.shape, 'but is:', magnitude.shape)
    #         return False
    # except Exception:
    #     print(traceback.format_exc())
    #     return False

    print ('=== Section 3.1 ===\n')
    print ('derivative using convolution')
    sys.stdout.flush()
    try:
        magnitude = sol2.conv_der(im)
        if not np.all(magnitude.shape==im.shape):
            print ('derivative magnitude shape should be :', im.shape, 'but is:' , magnitude.shape)
            return False
    except:
        print(traceback.format_exc())
        return False

    print ('=== Section 3.2 ===\n')
    print ('derivative using convolution')
    sys.stdout.flush()
    try:
        magnitude = sol2.fourier_der(im)
        if not np.all(magnitude.shape==im.shape):
            print ('derivative magnitude shape should be :', im.shape, 'but is:' , magnitude.shape)
            return False
    except:
        print(traceback.format_exc())
        return False

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir',
        default=None,
        nargs='?',
        help='Dummy argument for working with the CS testing system. Has no effect.'
    )
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Development mode: Assumes all student files are '
             'already under the directory "./current/"'
    )
    args = parser.parse_args()
    if not args.dev:
        try:
            shutil.rmtree('current')
            shutil.rmtree('current_tmp')
        except:
            pass
        os.makedirs('current_tmp')
        subprocess.check_call(['tar', 'xvf', sys.argv[1], '-C', 'current_tmp/'])
        os.rename('current_tmp/ex2','current')
    if not os.path.isfile('current/__init__.py'):
        with open('current/__init__.py', 'w') as f:
            f.write(' ')
    ### Supress matplotlib figures if display not available ###
    if os.getenv('DISPLAY') is None or os.getenv('DISPLAY') == '':
        import matplotlib
        matplotlib.use('PS')
    ###########
    if not presubmit():
        print('\n\n\n !!!!!!! === Presubmission Failed === !!!!!!! ')
    else:
        print('\n\n=== Presubmission Completed Successfully ===')
    print ("""\n
    Please go over the output and verify that there were no failures / warnings.
    Remember that this script tested only some basic technical aspects of your implementation.
    It is your responsibility to make sure your results are actually correct and not only
    technically valid.""")

