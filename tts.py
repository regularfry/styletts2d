import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import random
import yaml
import numpy as np
import torch
import torchaudio
import librosa
from nltk.tokenize import word_tokenize

from models import *
from utils import *
from text_utils import TextCleaner

import simpleaudio

import phonemizer

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

from Utils.PLBERT.util import load_plbert

from collections import OrderedDict

def set_espeak_path(lib_path):
    phonemizer.backend.espeak.wrapper.EspeakWrapper.set_library(lib_path)

class EspeakNotFoundError(Exception):
    pass

def get_espeak_backend(**kwargs):
    try:
        return phonemizer.backend.EspeakBackend(**kwargs)
    except RuntimeError as e:
        if e.args[0] == "espeak not installed on your system":
            raise EspeakNotFoundError()
        else:
            raise e

class TTS:
    def __init__(self):
        self.textcleaner = TextCleaner()
        self.to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80,
                                                           n_fft=2048,
                                                           win_length=1200,
                                                           hop_length=300)
        self.mean, self.std = -4, 4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.global_phonemizer = get_espeak_backend(language='en-us',
                                                    preserve_punctuation=True,
                                                    with_stress=True)
        with open("Models/LibriTTS/config.yml") as f:
            config = yaml.safe_load(f)

        # load pretrained ASR model
        ASR_config = config.get('ASR_config', False)
        ASR_path = config.get('ASR_path', False)
        self.text_aligner = load_ASR_models(ASR_path, ASR_config)

        # load pretrained F0 model
        F0_path = config.get('F0_path', False)
        self.pitch_extractor = load_F0_models(F0_path)

        # load BERT model
        BERT_path = config.get('PLBERT_dir', False)
        self.plbert = load_plbert(BERT_path)

        self.model_params = recursive_munch(config['model_params'])
        self.model = build_model(self.model_params, self.text_aligner, self.pitch_extractor, self.plbert)
        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]

        self.params_whole = torch.load("Models/LibriTTS/epochs_2nd_00020.pth", map_location='cpu')
        self.params = self.params_whole['net']

        for key in self.model:
            if key in self.params:
                print('%s loaded' % key)
                try:
                    self.model[key].load_state_dict(self.params[key])
                except:
                    state_dict = self.params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        _ = [self.model[key].eval() for key in self.model]

        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
            clamp=False
        )


    def _length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def _preprocess(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def compute_style(self, path):
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = self._preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)

    def inference(self, text, ref_s, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        tokens = self.textcleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = self._length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                                embedding=bert_dur,
                                embedding_scale=embedding_scale,
                                features=ref_s, # reference from the same speaker as the embedding
                                num_steps=diffusion_steps).squeeze(1)


            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en,
                                                s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)


            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr,
                                    F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later

    def make_sound(self, text, style, alpha=0.1, beta=0.1, steps=5, expressiveness=1):
        return self.inference(text, style, alpha=alpha, beta=beta, diffusion_steps=steps, embedding_scale=expressiveness)


def playback_wav(wav, rate):
    audio_array = (wav * 32767 / max(abs(wav))).astype(np.int16)
    playback = simpleaudio.play_buffer(audio_array, 1, 2, rate)
    playback.wait_done()


class Voice:
    def __init__(self, reference_wav):
        self.timbre_match_factor = 0.01
        self.prosody_match_factor = 0.01
        self.expressiveness = 2
        self.tts = TTS()
        self.style = self.tts.compute_style(reference_wav)

    def say(self, text):
        sound = self.tts.make_sound(text, self.style,
                                    alpha=self.timbre_match_factor,
                                    beta=self.prosody_match_factor,
                                    expressiveness=self.expressiveness)
        playback_wav(sound, 24000)
