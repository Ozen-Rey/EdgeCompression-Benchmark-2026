import visqol_lib_py  # type: ignore
from visqol.pb2 import visqol_config_pb2


def measure_visqol(reference_wav_path, degraded_wav_path):
    config = visqol_config_pb2.VisqolConfig()

    config.audio.sample_rate = 16000  # type: ignore
    config.options.use_speech_scoring = True  # type: ignore

    api = visqol_lib_py.VisqolApi()  # type: ignore
    api.Create(config)  # type: ignore

    # similarity_result = api.Measure(reference_wav_path, degraded_wav_path)  # type: ignore
    # return similarity_result.moslqo  # type: ignore


print("ViSQOL importato e caricato! Il C++ comunica con Python! 🎉")
