import numpy as np
from matplotlib import pyplot as plt
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from time import time


def load_hologram(path1, path2):
    img_ampli = cv2.imread(path1)
    img_phase = cv2.imread(path2)
    ampli = (img_ampli / 255)
    phase = 2.0 * np.pi * (img_phase / 255)
    hologram = ampli * np.exp(1j * phase)
    return hologram


def propASM(u1, pitch, lbd, z):
    M, N = u1.shape
    fx = np.arange(-1 / (2 * pitch), 1 / (2 * pitch), 1 / (N * pitch))
    fy = np.arange(-1 / (2 * pitch), 1 / (2 * pitch), 1 / (M * pitch))
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    w = np.maximum(0.0, (1 / lbd) ** 2 - FX ** 2 - FY ** 2) ** 0.5
    H = np.exp(-2j * np.pi * w * z)
    u2 = np.fft.fftshift(np.fft.fft2(u1))
    u2 = u2 * H
    u2 = np.fft.ifft2(np.fft.fftshift(u2))
    return u2


def Gera_Vista(holo, path_out, lin1=0, lin2=2048, col1=0, col2=2048, lbdr=6.4e-07, lbdg=5.32e-07, lbdb=4.73e-07, pitch=0.40e-06, z=0.0031):
    holo1 = np.zeros(holo.shape, dtype=holo.dtype)
    holo1[lin1:lin2, col1:col2, :] = holo[lin1:lin2, col1:col2, :]
    recons = np.zeros(holo.shape, dtype=holo.dtype)
    recons[:, :, 0] = propASM(holo1[:, :, 2], pitch, lbdr, z)
    recons[:, :, 1] = propASM(holo1[:, :, 1], pitch, lbdg, z)
    recons[:, :, 2] = propASM(holo1[:, :, 0], pitch, lbdb, z)
    arecons = np.abs(recons)
    arecons = (arecons * 255).astype('uint8')
    filename = path_out + str(lin1) + '_' + str(col1) + '_' + str(int(100000 * z)) + '.png'
    plt.imsave(filename, arecons)
    return recons


def Gera_Vistas(holo, path_out, lbdr=6.4e-07, lbdg=5.32e-07, lbdb=4.73e-07, pitch=0.4e-06, z=0.0031):
    for col1 in (0, 1024, 2048):
        for lin1 in (0, 1024, 2048):
            col2 = 2048 + col1
            lin2 = 2048 + lin1
            holo1 = np.zeros(holo.shape, dtype=holo.dtype)
            holo1[lin1:lin2, col1:col2, :] = holo[lin1:lin2, col1:col2, :]
            t1 = time()
            recons = np.zeros(holo.shape, dtype=holo.dtype)
            recons[:, :, 0] = propASM(holo1[:, :, 2], pitch, lbdr, z)
            recons[:, :, 1] = propASM(holo1[:, :, 1], pitch, lbdg, z)
            recons[:, :, 2] = propASM(holo1[:, :, 0], pitch, lbdb, z)
            t2 = time()
            print('Reconstruction time: %.3f' % (t2 - t1))
            recons = np.abs(recons)
            recons = (recons * 255).astype('uint8')
            filename = path_out + str(lin1) + '_' + str(col1) + '_' + str(int(100000 * z)) + '.png'
            plt.imsave(filename, recons)


def ViewVista(im1, im2):
    fig = plt.figure(figsize=(30,15))

    plt.subplot(121)
    plt.axis('off')
    plt.imshow(im1)

    plt.subplot(122)
    plt.axis('off')
    plt.imshow(im2)

    plt.show()


def main(path1='/OCaml/home/Reaserch/biplane8k_ampli.png', path2='/OCaml/home/Reaserch/biplane8k_phase.png', path_out='dices_test1', lbdr=640e-9, lbdg=532e-9, lbdb=473e-9, pitch=1e-6, z=2.175e-2):
    holo = load_hologram(path1, path2)

    reconst_dir = 'Dices_Teste' + path1[0:len(path1) - 10].replace(':', '')
    if not os.path.exists(reconst_dir):
        os.makedirs(reconst_dir)

    for i in range(2):
        for j in range(2):
            lin1 = i * 2048
            lin2 = (i + 1) * 2048
            col1 = j * 2048
            col2 = (j + 1) * 2048

            recons = Gera_Vista(holo, reconst_dir + '/' + path_out + f'{i}{j}', lin1, lin2, col1, col2, lbdr, lbdg, lbdb, pitch, z)
            Arecons = np.abs(recons)
            Phrecons = np.angle(recons)
            ViewVista(Arecons, Phrecons)

main()

# -C K, --modular_colorspace=K
#    Color transform: -1 = default (try several per group, depending
#    on effort), 0 = RGB (none), 1-41 = fixed RCT (6 = YCoCg)


#Ver a alteraçao que o jpegxl irá fazer alteracao (flotoante) Documento
#PFM
#