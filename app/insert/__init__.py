from pathlib import Path
from app.insert.insert import Insert
from config import Config


encoderPath=Config.INSERT_DIR+"encoder/saved_models/pretrained.pt"
synthesizerPath=Path(Config.INSERT_DIR+"lib/synthesizer/saved_models/logs-pretrained")
vocoderPath=Config.INSERT_DIR+"lib/vocoder/saved_models/pretrained/pretrained.pt"

insert_object=Insert(encoderModelPath=encoderPath,synthesizerModelPath=synthesizerPath,vocoderModelPath=vocoderPath)
