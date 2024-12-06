import torch
import numpy as np
import librosa
from scipy.io.wavfile import write

import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence
from text.symbols import symbols


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    return torch.LongTensor(text_norm)


def main():
    config_path = "./configs/swara_config.json"  #modify here
    checkpoint_path = "./logs/vits1/G_261500.pth"  # here 2
    output_wav_path = "./output_2.wav"

    hps = utils.get_hparams_from_file(config_path)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).cuda()
    _ = net_g.eval()

    utils.load_checkpoint(checkpoint_path, net_g, None)

    text = "test casa test cana"
    stn_tst = get_text(text, hps)

    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(
            x_tst,
            x_tst_lengths,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1.0
        )[0][0, 0].data.cpu().numpy()

    audio = (audio * 32768).astype(np.int16)
    write(output_wav_path, hps.data.sampling_rate, audio)


if __name__ == "__main__":
    main()
