
python /Users/didac/proj/didac/code/forecasting/scripts/run_lta.py \
--job_name slowfast_recognition_eval
--working_directory checkpoints/recognition/
--cfg /proj/vondrick/didac/code/forecasting/configs/Ego4dRecognition/MULTISLOWFAST_8x8_R101.yaml
DATA.PATH_TO_DATA_DIR /proj/vondrick/didac/code/forecasting/data/long_term_anticipation/annotations/
DATA.PATH_PREFIX /proj/vondrick/didac/code/forecasting/data/long_term_anticipation/clips/
CHECKPOINT_LOAD_MODEL_HEAD True
MODEL.FREEZE_BACKBONE False
DATA.CHECKPOINT_MODULE_FILE_PATH ""
#CHECKPOINT_FILE_PATH /proj/vondrick/didac/code/forecasting/pretrained_models/long_term_anticipation/kinetics_slowfast8x8.ckpt
CHECKPOINT_FILE_PATH /proj/vondrick/didac/code/forecasting/pretrained_models/long_term_anticipation/ego4d_slowfast8x8.ckpt
TRAIN.ENABLE False
TRAIN.BATCH_SIZE 8
NUM_GPUS 1