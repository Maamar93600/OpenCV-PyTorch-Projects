# I took your code and tried to understand it; personally, I don't like copy-pasting without understanding.
from librairie import *

def rle_encode(mask):
    pixels = mask.flatten(order='F')  # match the competition's order

    pixels = np.concatenate([[0], pixels,
                             [0]])  # pour garantir une transition debut 0->1 ou fin 1-> 0, ajout sur axe =0 car tous est aplatie

    runs = np.where(pixels[1:] != pixels[:-1])[
               0] + 1  # et donne les positions de transitions entre 0 et 1 décalage utilisé par RLE

    runs[1::2] -= runs[
        ::2]  # dans les positions impair , on remplace par la longueur donc "RUN_TEST[1::2]=RUN_TEST[1::2] - RUN_TEST[::2]" Fin - Debut

    return ' '.join(str(x) for x in runs)
