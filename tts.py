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
        neural_factory = nemo.core.NeuralModuleFactory(placement=device)

        # Create text to spectrogram model
        tacotron2_params = parse_yaml("./NeMo/examples/tts/configs/tacotron2.yaml")
        self.tts_conf = parse_yaml("conf.yaml")["tts"]

        # create preprocessor
        data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
                **tacotron2_params["AudioToMelSpectrogramPreprocessor"])
        text_embedding = nemo_tts.TextEmbedding(
            len(tacotron2_params["labels"]) + 3,  # + 3 special chars
            **tacotron2_params["TextEmbedding"])
        t2_enc = nemo_tts.Tacotron2Encoder(**tacotron2_params["Tacotron2Encoder"])
        t2_dec = nemo_tts.Tacotron2DecoderInfer(
                **tacotron2_params["Tacotron2Decoder"])
        t2_postnet = nemo_tts.Tacotron2Postnet(
            **tacotron2_params["Tacotron2Postnet"])
        t2_loss = nemo_tts.Tacotron2Loss(**tacotron2_params["Tacotron2Loss"])
        makegatetarget = nemo_tts.MakeGate()

        neural_factory.logger.info('================================')
        neural_factory.logger.info(
            f"Number of parameters in text embedding: {text_embedding.num_weights}")
        neural_factory.logger.info(
            f"Number of parameters in encoder: {t2_enc.num_weights}")
        neural_factory.logger.info(
            f"Number of parameters in decoder: {t2_dec.num_weights}")
        neural_factory.logger.info(
            f"Number of parameters in postnet: {t2_postnet.num_weights}")
        neural_factory.logger.info(
            f"Total number of parameters in model: \
            {text_embedding.num_weights + t2_enc.num_weights + t2_dec.num_weights + t2_postnet.num_weights}")
        neural_factory.logger.info('================================')

        spec_neural_modules = ( data_preprocessor, text_embedding, t2_enc, t2_dec,
                                t2_postnet, t2_loss, makegatetarget)

    
    def synthesis(self, text):
        
        # create inference DAGs
        data_layer = nemo_asr.TranscriptDataLayer(
            path="text.json",
            labels=tacotron2_params['labels'],
            batch_size=1,
            num_workers=1,
            load_audio=False,
            bos_id=len(tacotron2_params['labels']),
            eos_id=len(tacotron2_params['labels']) + 1,
            pad_id=len(tacotron2_params['labels']) + 2,
            shuffle=False
        )
        transcript, transcript_len = data_layer()

        transcript_embedded = text_embedding(char_phone=transcript)
        transcript_encoded = t2_enc(
            char_phone_embeddings=transcript_embedded,
            embedding_length=transcript_len)
        mel_decoder, gate, alignments, mel_len = t2_dec(
            char_phone_encoded=transcript_encoded,
            encoded_length=transcript_len)

        mel_postnet = t2_postnet(mel_input=mel_decoder)
        infer_tensors = [mel_postnet, gate, alignments, mel_len]

        # Run tacotron 2
        neural_factory.logger.info("Running Tacotron 2")
        evaluated_tensors = neural_factory.infer(
            tensors=infer_tensors,
            checkpoint_dir=tts_conf["model_dir"],
            cache=True,
            offload_to_cpu=True ##check this
        )
        mel_len = evaluated_tensors[-1]
        neural_factory.logger.info("Done Running Tacotron 2")

        filterbank = librosa.filters.mel(
            sr=tacotron2_params["sample_rate"],
            n_fft=tacotron2_params["n_fft"],
            n_mels=tacotron2_params["n_mels"],
            fmax=tacotron2_params["fmax"])



        if tts_conf["vocoder"] == "griffin-lim":
            n_iters = 50
            n_fft = 1024
            griffin_lim_power = 1.2
            griffin_lim_mag_scale = 2048
            neural_factory.logger.info("Running Griffin-Lim as a vocoder")
            mel_spec = evaluated_tensors[0][0]
            log_mel = mel_spec.cpu().numpy().transpose(0, 2, 1)
            mel = np.exp(log_mel)
            sample = np.dot(mel, filterbank) * griffin_lim_mag_scale
            sample = sample[0][:mel_len[0][0], :]

            # convert magnitude spectrograms to audio signal
            magnitudes = sample.T ** griffin_lim_power
            phase = np.exp(2j * np.pi * np.random.rand(*magnitudes.shape))
            complex_spec = magnitudes * phase
            signal = librosa.istft(complex_spec)
            if np.isfinite(signal).all():
                for _ in range(n_iters):
                    _, phase = librosa.magphase(librosa.stft(signal, n_fft=n_fft))
                    complex_spec = magnitudes * phase
                    signal = librosa.istft(complex_spec)
            else:
                neural_factory.logger.warn("audio was not finite")
                signal = np.array([0])
            save_file = "griffin_sample.wav"
            wavfile.write(save_file, tacotron2_params["sample_rate"], signal)
            neural_factory.logger.info("Wav file was generated and named: "+save_file)
        else:
            waveglow_sigma = self.tts_conf["sigma"]
            waveglow_denoiser_strength = self.tts_conf["denoising_strength"]
            neural_factory.logger.info("Running waveglow as a vocoder")
            waveglow_params = parse_yaml("./NeMo/examples/tts/configs/waveglow.yaml")
            waveglow = nemo_tts.WaveGlowInferNM(
                sigma=waveglow_sigma,
                **waveglow_params["WaveGlowNM"])
            mel_pred = infer_tensors[0]
            audio_pred = waveglow(mel_spectrogram=mel_pred)

            # Run waveglow
            neural_factory.logger.info("Running Waveglow")
            evaluated_tensors = neural_factory.infer(
                tensors=[audio_pred],
                checkpoint_dir=tts_conf["vocoder_dir"],
                modules_to_restore=[waveglow],
                use_cache=True
            )
            neural_factory.logger.info("Done Running Waveglow")

            if waveglow_denoiser_strength > 0:
                neural_factory.logger.info("Setup denoiser")
                waveglow.setup_denoiser()

            neural_factory.logger.info("Saving results to disk")
            mel_spec = evaluated_tensors[0][0]
            sample = mel_spec.cpu().numpy()[0]
            sample_len = mel_len[0][0] * tacotron2_params["n_stride"]
            sample = sample[:sample_len]
            # apply denoiser
            if waveglow_denoiser_strength > 0:
                sample, spec = waveglow.denoise(sample,
                                    strength = waveglow_denoiser_strength)
            else:
                spec, _ = librosa.core.magphase(librosa.core.stft(
                    sample, n_fft = waveglow_params["n_fft"]))
            save_file = "waveglow_sample.wav"
            wavfile.write(save_file, waveglow_params["sample_rate"], sample)
