import numpy as np
from imageio import imread
from scipy.signal import convolve2d
from skimage.color import rgb2gray
import scipy.io.wavfile as wv
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates

GRAYSCALE_REP = 1
RGB_REP = 2
GRAY_MAX_LEVEL = 255
CONV_ROW_VECTOR = np.array([[-0.5, 0, 0.5]])
CONV_COLUMN_VECTOR = np.transpose(CONV_ROW_VECTOR)


def read_image(filename, representation):
    # reads image from filename returns its array rep

    img = np.float64(imread(filename))/GRAY_MAX_LEVEL

    if representation == GRAYSCALE_REP:
        img = rgb2gray(img)

    return np.float64(img)


def compute_dft_matrix(N):
    # returns dft matrix

    power = ((-2 * np.pi * 1j) / N) * np.arange(N)
    power = power * np.arange(N)[:, np.newaxis]
    dft_matrix = np.exp(power)
    return dft_matrix


def DFT(signal):
    # transform 1-D signal to fourier signal

    N = signal.shape[0]
    dft_matrix = compute_dft_matrix(N)
    fourier_signal = np.dot(dft_matrix, signal)
    return fourier_signal


def IDFT(fourier_signal):
    # transform 1-D fourier signal to complex signal

    N = fourier_signal.shape[0]
    dft_matrix = np.linalg.inv(compute_dft_matrix(N))
    try:
        signal = np.dot(fourier_signal, dft_matrix)
    except ValueError:
        signal = np.dot(dft_matrix, fourier_signal)
    return signal


def DFT2(image):
    # transform 2-D signal to 2-D fourier signal

    temp = np.transpose(DFT(image))
    return np.transpose(DFT(temp))


def IDFT2(fourier_image):
    # transform 2-D fourier signal to 2-D complex signal

    temp = np.transpose(IDFT(fourier_image))
    return np.transpose(IDFT(temp))


def change_rate(filename, ratio):
    # changing the sample rate by ratio

    sr, data = wv.read(filename)
    wv.write("change_rate.wav", int(sr*ratio), data)


def zero_padding(data, new_size, ratio):
    # creates larger array with 0 padding the edges

    N = data.shape[0]
    # creating zero complex array of the desired size
    zero_arr = np.zeros(new_size, dtype=np.complex128)
    print(zero_arr)
    #starting index
    f_index = int((new_size-N)/2)
    print(f_index)
    zero_arr[f_index:f_index+N] = data
    return zero_arr


def clip_hfreq(data, new_size):
    # creates new smaller array after clipping high values

    N = data.shape[0]
    # starting index
    f_index = int((N-new_size)/2)
    # slicing arrayto get desired size
    new_data = data[f_index:f_index+new_size]
    return new_data


def resize(data, ratio):
    # resize array shape according to given ratio

    N = data.shape[0]
    fourier_series = DFT(data)
    # shifting
    shifted = np.fft.fftshift(fourier_series)
    new_size = int(np.floor(N/ratio))
    if ratio <= 1:
        # larger array -> pdding with zero
        edited_arr = zero_padding(shifted, new_size, ratio)
    else:
        # smaller array clip high freq
        edited_arr = clip_hfreq(shifted, new_size)

    # shifting back
    shifted_back = np.fft.ifftshift(edited_arr)
    new_data = IDFT(shifted_back)

    # if data.dtype was float 64
    if new_data.dtype != data.dtype:
        new_data = new_data.real
        new_data.astype(np.float64)

    return new_data


def change_samples(filename, ratio):
    # change the duration of audio file using fourier

    sr, data = wv.read(filename)
    new_data = resize(data, ratio)
    new_data = new_data.real

    return new_data.astype(np.float64)


def resize_spectrogram(data, ratio):
    # resize spectrogram using stft,istft and resizing along axis by given ratio

    stft_arr = stft(data)
    # resizing
    updated_columns = np.apply_along_axis(resize, 1, arr=stft_arr, ratio=ratio)
    new_data = istft(updated_columns)

    return new_data.astype(np.float64)


def resize_vocoder(data, ratio):
    # resize using vocoder func by given ratio

    arr = stft(data)
    new_data = phase_vocoder(arr, ratio)
    new_data = istft(new_data)

    return new_data.astype(np.float64)


def conv_der(im):
    # Convolution derivative using [-0.5,0,0.5] vector

    dx = convolve2d(im, CONV_ROW_VECTOR, mode="same")
    dy = convolve2d(im, CONV_COLUMN_VECTOR, mode="same")

    magnitude = np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)

    return magnitude


def fourier_der(im):
    # derivatives using fourier transform
    fourier_im = DFT2(im)
    shifted = np.fft.fftshift(fourier_im)

    x_dim = im.shape[0]
    y_dim = im.shape[1]
    #compute u ,v vectors
    u_arr = np.arange(-(x_dim//2), x_dim//2)
    v_arr = np.arange(-(y_dim//2), y_dim//2)
    # derivatives
    der_by_u = np.transpose(shifted) * u_arr
    der_by_v = shifted * v_arr
    dx = IDFT2(np.transpose(der_by_u))
    dy = IDFT2(der_by_v)

    magnitude = np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)

    return magnitude


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec

