import os
import nemo
import librosa
import nemo_asr
import nemo_tts
import numpy as np
from scipy.io import wavfile

from utils import *



class TTS:
    def __init__(self):
        device = nemo.core.DeviceType.CPU
        self.nf = nemo.core.NeuralModuleFactory(placement=device)

        # Create text to spectrogram model
        self.tts_conf = parse_yaml("conf.yaml")["tts"]
        self.tacotron2_params = parse_yaml("./NeMo/examples/tts/configs/tacotron2.yaml")

        self.nf.logger.info('================================')
        # create text embedding module
        self.text_embedding = nemo_tts.TextEmbedding(
                len(self.tacotron2_params["labels"]) + 3, # + 3 special chars
                **self.tacotron2_params["TextEmbedding"])
        self.nf.logger.info(f"Number of parameters in text-embedding: {self.text_embedding.num_weights}")
        
        # create encoder
        self.t2_enc = nemo_tts.Tacotron2Encoder(
                    **self.tacotron2_params["Tacotron2Encoder"])
        self.nf.logger.info(
            f"Number of parameters in encoder: {self.t2_enc.num_weights}")
        
        # create decoder
        self.t2_dec = nemo_tts.Tacotron2DecoderInfer(
                    **self.tacotron2_params["Tacotron2Decoder"])
        self.nf.logger.info(
            f"Number of parameters in decoder: {self.t2_dec.num_weights}")
        
        # create PostNet
        self.t2_postnet = nemo_tts.Tacotron2Postnet(
                    **self.tacotron2_params["Tacotron2Postnet"])
        self.nf.logger.info(
            f"Number of parameters in postnet: {self.t2_postnet.num_weights}")
        total_weights= self.text_embedding.num_weights+self.t2_enc.num_weights \
                        + self.t2_dec.num_weights + self.t2_postnet.num_weights
        
        self.nf.logger.info(
            f"Total number of parameters in model: {total_weights}")
        self.nf.logger.info('================================')

        # load waveglow if chosen
        if self.tts_conf["vocoder"] == "waveglow":
            self.nf.logger.info("Running waveglow as a vocoder")
            self.waveglow_params = \
                    parse_yaml("./NeMo/examples/tts/configs/waveglow.yaml")
            self.waveglow = nemo_tts.WaveGlowInferNM(
                                    sigma = self.tts_conf["sigma"],
                                    **self.waveglow_params["WaveGlowNM"])
            if self.tts_conf["denoising_strength"] > 0:
                self.nf.logger.info("Setup denoiser for waveglow")
                self.waveglow.setup_denoiser()
                self.nf.logger.info("Waveglow denoiser is ready")

    
    def synthesis(self, text):
        self.nf.logger.info('================================')
        self.nf.logger.info('Starting speech synthesis')
        # create inference DAGs
        data_layer = nemo_asr.TranscriptDataLayer(
            path = build_text_path(text),
            labels = self.tacotron2_params['labels'],
            batch_size = 1,
            num_workers = 1,
            load_audio=False,
            bos_id = len(self.tacotron2_params['labels']),
            eos_id = len(self.tacotron2_params['labels']) + 1,
            pad_id = len(self.tacotron2_params['labels']) + 2,
            shuffle = False
        )
        os.remove("text.json")
        self.nf.logger.info("Running Tacotron 2")
        transcript, transcript_len = data_layer()

        transcript_embedded = self.text_embedding(char_phone=transcript)
        transcript_encoded = self.t2_enc(
                                char_phone_embeddings=transcript_embedded,
                                embedding_length=transcript_len)
        mel_decoder, gate, alignments, mel_len = self.t2_dec(
                                char_phone_encoded=transcript_encoded,
                                encoded_length=transcript_len)
        mel_postnet = self.t2_postnet(mel_input=mel_decoder)
        infer_tensors = [mel_postnet, gate, alignments, mel_len]
        # Run tacotron 2
        evaluated_tensors = self.nf.infer(
                                    tensors = infer_tensors,
                                    checkpoint_dir = self.tts_conf["model_dir"],
                                    cache = True,
                                    offload_to_cpu = True)
        mel_len = evaluated_tensors[-1]

        # creating vocoder
        if self.tts_conf["vocoder"] == "griffin-lim":
            self.nf.logger.info("Running Griffin-Lim as a vocoder")
            mel_spec = evaluated_tensors[0][0]
            log_mel = mel_spec.cpu().numpy().transpose(0, 2, 1)
            mel = np.exp(log_mel)
            filterbank = librosa.filters.mel(
                                sr = self.tacotron2_params["sample_rate"],
                                n_fft = self.tacotron2_params["n_fft"],
                                n_mels = self.tacotron2_params["n_mels"],
                                fmax = self.tacotron2_params["fmax"])
            sample = np.dot(mel, filterbank) * self.tts_conf["mag_scale"]
            sample = sample[0][:mel_len[0][0], :]

            # convert magnitude spectrograms to audio signal
            magnitudes = sample.T ** self.tts_conf["power"]
            phase = np.exp(2j * np.pi * np.random.rand(*magnitudes.shape))
            complex_spec = magnitudes * phase
            signal = librosa.istft(complex_spec)
            if np.isfinite(signal).all():
                for _ in range(self.tts_conf["n_iters"]):
                    _, phase = librosa.magphase(librosa.stft(signal,
                                    n_fft = self.tts_conf["n_fft"]))
                    complex_spec = magnitudes * phase
                    signal = librosa.istft(complex_spec)
            else:
                self.nf.logger.warn("audio was not finite")
                signal = np.array([0])
            save_file = "griffin_sample.wav"
            wavfile.write(save_file, self.tacotron2_params["sample_rate"], signal)
            self.nf.logger.info("Wav file was generated and named: "+save_file)
        
        elif self.tts_conf["vocoder"] == "waveglow":
            self.nf.logger.info("Running Waveglow as a vocoder")
            audio_pred = self.waveglow(mel_spectrogram=mel_postnet)
            # Run waveglow
            evaluated_tensors = self.nf.infer(
                tensors = [audio_pred],
                checkpoint_dir = self.tts_conf["vocoder_dir"],
                modules_to_restore = [self.waveglow],
                use_cache = True
            )
            self.nf.logger.info("Done Running Waveglow")
            mel_spec = evaluated_tensors[0][0]
            sample = mel_spec.cpu().numpy()[0]
            sample_len = mel_len[0][0] * self.tacotron2_params["n_stride"]
            sample = sample[:sample_len]
            
            # apply denoiser
            waveglow_denoiser_strength = self.tts_conf["denoising_strength"]
            if waveglow_denoiser_strength > 0:
                sample, spec = self.waveglow.denoise(sample,
                                    strength = waveglow_denoiser_strength)
            else:
                spec, _ = librosa.core.magphase(librosa.core.stft(
                             sample, n_fft = self.waveglow_params["n_fft"]))
            save_file = "waveglow_sample.wav"
            wavfile.write(save_file, self.waveglow_params["sample_rate"], sample)
            self.nf.logger.info("Wav file was generated and named: "+save_file)



if __name__ == "__main__":
    tts = TTS()
    tts.synthesis("Abdo is here.")
    # this will generate a wav file in the current directory 
