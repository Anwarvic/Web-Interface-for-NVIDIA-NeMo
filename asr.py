import os
import copy

import nemo
import nemo_asr
from nemo_asr.helpers import post_process_predictions
from utils import *


class ASR:
    def __init__(self):
        """Loads pre-trained ASR model"""
        self.asr_conf = parse_yaml()["asr"]
        device = nemo.core.DeviceType.CPU
        self.nf = nemo.core.NeuralModuleFactory(placement=device)
        # load model configuration
        jasper_params = parse_yaml(
            os.path.join(self.asr_conf["model_dir"], "quartznet15x5.yaml"))
        self.labels = jasper_params["labels"]
        self.sample_rate = jasper_params["sample_rate"]

        # preprocessor
        self.eval_dl_params = copy.deepcopy(jasper_params["AudioToTextDataLayer"])
        self.eval_dl_params.update(jasper_params["AudioToTextDataLayer"]["eval"])
        del self.eval_dl_params["train"]
        del self.eval_dl_params["eval"]
        self.preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
            sample_rate = self.sample_rate,
            **jasper_params["AudioPreprocessing"])
        
        # model encoder
        feats = jasper_params["AudioPreprocessing"]["features"]
        self.jasper_encoder = nemo_asr.JasperEncoder(
            feat_in = feats,
            **jasper_params["JasperEncoder"])
        self.jasper_encoder.restore_from(
                            os.path.join(self.asr_conf["model_dir"],
                                        "JasperEncoder-STEP-247400.pt"))

        # model decoder
        filters = jasper_params["JasperEncoder"]["jasper"][-1]["filters"]
        self.jasper_decoder = nemo_asr.JasperDecoderForCTC(
            feat_in = filters,
            num_classes=len(self.labels))
        self.jasper_decoder.restore_from(
                            os.path.join(self.asr_conf["model_dir"],
                                        "JasperDecoderForCTC-STEP-247400.pt"))

        self.nf.logger.info('================================')
        self.nf.logger.info(
            f"Number of parameters in encoder: {self.jasper_encoder.num_weights}")
        self.nf.logger.info(
            f"Number of parameters in decoder: {self.jasper_decoder.num_weights}")
        self.nf.logger.info(
            f"Total number of parameters in model: "
            f"{self.jasper_decoder.num_weights + self.jasper_encoder.num_weights}")
        self.nf.logger.info('================================')
        
        # CTC decoder
        if self.asr_conf["decoder"] == "beam":
            self.ctc_decoder = nemo_asr.BeamSearchDecoderWithLM(
                    vocab = self.labels,
                    beam_width = self.asr_conf["beam_width"],
                    alpha = self.asr_conf["alpha"],
                    beta = self.asr_conf["beta"],
                    lm_path = self.asr_conf["lm_path"],
                    num_cpus = max(os.cpu_count(), 1))
        else:
            self.ctc_decoder = nemo_asr.GreedyCTCDecoder()



    def transcribe(self, wav_path):
        """Reads audio file and returns the recognized transcrition"""
        self.nf.logger.info('Started Transcribing Speech')
        data_layer = nemo_asr.AudioToTextDataLayer(
            manifest_filepath = build_manifest(wav_path),
            sample_rate = self.sample_rate,
            labels = self.labels,
            batch_size = 1,
            **self.eval_dl_params)
        os.remove("audio.json")
        self.nf.logger.info('Loading {0} examples'.format(len(data_layer)))

        audio_sig_e1, a_sig_length_e1, transcript_e1, transcript_len_e1 = data_layer()

        # apply pre-processing 
        processed_signal_e1, p_length_e1 = self.preprocessor(
            input_signal = audio_sig_e1,
            length = a_sig_length_e1)

        # encode audio signal
        encoded_e1, encoded_len_e1 = self.jasper_encoder(
            audio_signal=processed_signal_e1,
            length=p_length_e1)

        # decode encoded signal
        log_probs_e1 = self.jasper_decoder(encoder_output=encoded_e1)

        # apply CTC decode
        if self.asr_conf["decoder"] == "beam":
            beam_predictions_e1 = self.ctc_decoder(
                    log_probs=log_probs_e1, log_probs_length=encoded_len_e1)
            evaluated_tensors = self.nf.infer(
                    tensors=[beam_predictions_e1],
                    use_cache=False)
            hypotheses = []
            # Over mini-batch
            for i in evaluated_tensors[1]:
                hypotheses.append(i)
        else:
            greedy_predictions_e1 = self.ctc_decoder(log_probs=log_probs_e1)
            eval_tensors = [log_probs_e1, greedy_predictions_e1,
                            transcript_e1, transcript_len_e1, encoded_len_e1]
            evaluated_tensors = self.nf.infer(
                tensors = eval_tensors,
                cache = True
            )
            hypotheses = post_process_predictions(
                evaluated_tensors[1],
                self.labels)
        
        return hypotheses






if __name__ == "__main__":
    asr = ASR()
    wav_path = "romance_gt.wav"
    text = asr.transcribe(wav_path)
    print("You said:", text)